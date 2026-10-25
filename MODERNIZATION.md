# MM-OR modernization: changes and verification

Branch `modernize/torch29-cu128` ports the MM2SG scene-graph pipeline from its original
2023 stack to a current one, without changing what the model computes:

| | original (baseline) | modernized |
|---|---|---|
| Python | 3.10 | 3.12 |
| torch | 2.0.1 + cu118 | 2.9.1 + **cu128** (covers sm_86 … sm_120 / Blackwell) |
| transformers | 4.31.0 | 4.57.6 |
| numpy | 1.26 | 2.2.6 |
| flash-attn | 2.3.4 | 2.8.3 (release wheel `cu12torch2.9cxx11abiTRUE-cp312`) |
| spconv | `spconv-cu117` 2.3.6 | `spconv-cu126` 2.3.8 |
| torch-scatter | 2.1.2 | **dropped** (native ATen `segment_csr`) |
| open3d | 0.17 | **dropped** (built-in PCD reader) |

The source tree is **bi-compatible**: it still imports and runs on the original stack. That
is what allows the original environment to serve as a parity oracle for every change below.

**Result: the modernized stack reproduces the original pipeline's metrics on the full test
set (combined Macro-F1 0.691 vs 0.689, every take within ±0.007), and its training loss
tracks the original within run-to-run noise over 200 steps (see
[Verification](#3-verification)).**

---

## Installation

```bash
cd scene_graph_generation
./setup_env_modern.sh            # --help for options; --locked for the exact verified set
source .venv-mmor/bin/activate
```

Everything comes from prebuilt wheels, so no CUDA toolkit or compiler is needed. The
requirements are Linux x86_64, Python 3.12 and a driver for CUDA ≥ 12.8. The README's
[evaluation walkthrough](README.md#evaluating-the-published-checkpoints-step-by-step-modernized-stack)
covers data, checkpoints and the eval command.

| file | contents |
|---|---|
| `requirements-modern/torch.txt` | torch 2.9.1 / torchvision 0.24.1, installed from the cu128 index |
| `requirements-modern/requirements.txt` | pinned top-level packages (transformers 4.57.6, peft 0.21.2, bitsandbytes 0.50.2, spconv-cu126 2.3.8, …) |
| `requirements-modern/lock.txt` | `pip freeze` of the environment the metrics below were produced with |
| `setup_env_modern.sh` | installs the above, plus the flash-attn 2.8.3 release wheel, LLaVA (editable, `--no-deps`) and the PTv3 init weights |

`requirements.txt` and `LLaVA/pyproject.toml` are not used for this stack. The first still
pins the original 2023 versions for the baseline environment. The second deliberately
declares no versions.

**The install recipe is verified.** I built a fresh venv with `setup_env_modern.sh` in a
clean checkout of this branch and ran the README eval command on the released stage-1
checkpoint (`--limit-val-batches 20`). Every metric line was identical to the environment
behind the results in §3. That run also used a newly created
`pointtransformerv3_weights.pth`, which shows the file's random content does not matter
for evaluation: the checkpoint overwrites all PTv3 weights.

### Troubleshooting

| symptom | cause / fix |
|---|---|
| `AssertionError: No weights found at .../pointtransformerv3_weights.pth` | The PTv3 init file is required even for evaluation. Run `cd LLaVA && python -m llava.model.multimodal_projector.pointtransformerv3` once, on a GPU. |
| `RuntimeWarning: flash_attn not installed: PointTransformerV3 is falling back …` | flash-attn is missing, so results deviate from the verified numbers. Install the release wheel (see `setup_env_modern.sh`). Never `pip install flash-attn`: that compiles from source. |
| pip tries to downgrade torch, or installs transformers 4.31 / numpy 1.26 | You used `requirements.txt` or a plain `pip install -e LLaVA`. Use `requirements-modern/` and `pip install --no-deps -e LLaVA`. |
| `ModuleNotFoundError: torch_scatter` / `open3d` | Code from `master`, not this branch. Check out `modernize/torch29-cu128`. |
| `FileNotFoundError: .../mm_projector.bin` (or another loading error) | The checkpoint directory was renamed. The loader needs `llava` and `lora` in the directory name to load it as a LoRA adapter on top of `liuhaotian/llava-v1.5-7b`. |
| Metrics far below §3, quietly | Missing modality folders are skipped without an error (§2). Check that the per-timepoint audio, transcript and segmask folders exist for the test takes. |
| `CUDA driver version is insufficient` / `no kernel image is available` | The driver is too old for cu128 (`nvidia-smi` must report CUDA ≥ 12.8), or the GPU is older than the cu128 wheels support. Verified on A40 and RTX A6000 (sm_86). |

---

## 1. Code changes

The full rationale for each change is in its commit message. Summary:

### transformers 4.57 / torch 2.9 compatibility

- **`model_type` `llava` → `llava_mm2sg`.** transformers ships its own LLaVA under
  `"llava"`, so `AutoConfig.register` raised. Released checkpoints still declare `"llava"`
  and still load, because every load site now constructs `LlavaConfig` explicitly. Going
  through `AutoConfig` would return transformers' own config class and silently drop
  MM-OR's custom fields.
- **Cache API.** `llava_arch.py` read the KV-cache length with the pre-4.36 tuple layout.
  It now uses `Cache.get_seq_length()`, with the old indexing kept as a fallback.
- **`forward()` accepts `cache_position` and `**kwargs`**, which `generate()` passes since 4.38.
- **Batched generation with padding (`llava_llama.py`).** This was the most consequential
  fix, and it was invisible to every unit-level check. `generate()` derives
  `cache_position` from the text-only input length, but the KV cache holds the spliced
  sequence (text plus modality embeddings), which is longer. Forwarding it built a causal
  mask that hid most of the context from each new token. The effect showed only when an
  explicit mask was materialized, i.e. when a batch contained padding, and then **every
  row of that batch** came out as garbage:
  - 4D-OR prompts have a fixed length, so 4D-OR matched exactly.
  - MM-OR prompts vary in length (speech/metadata), so MM-OR fell to Macro-F1 0.106.
  - Fix: never forward `cache_position` when images are present.
- **QLoRA checkpoint loading (`builder.py`).** The released checkpoints carry their
  training-time bnb nf4 `quantization_config` in `config.json`.
  - transformers 4.31 ignored it, so **the original pipeline evaluates in bf16**.
  - 4.57 honors it and builds 4-bit layers that the bf16 base never fills, which fails
    with `Cannot copy out of meta tensor`.
  - It is now dropped unless `load_4bit`/`load_8bit` is requested, which preserves the
    original bf16 behaviour.
- **Trainer.** `ALL_LAYERNORM_LAYERS` moved to `transformers.pytorch_utils`. The
  `ShardedDDPOption` / fairscale branches were removed upstream and were never taken here,
  since training uses DeepSpeed ZeRO-2.
- **`torch.load`.** All 13 call sites now pass an explicit `weights_only=True`, because
  the default flipped in torch 2.6. Every site was audited against the released artifacts:
  all of them load tensors only.
- **`pyproject.toml`** no longer pins the 2023 stack. `pip install -e .` used to try to
  downgrade torch 2.9 → 2.0. The web-demo extras moved to an optional `demo` extra.

### Training path (transformers 4.57 / peft 0.21 / DeepSpeed 0.19)

- **Unfrozen CLIP layers received no gradient (silent).** transformers >= 4.35 enables
  gradient checkpointing in *every* submodule that supports it, including the nested CLIP
  vision encoder; 4.31 only checkpointed Llama. Under the default *reentrant* checkpointing,
  a block whose input does not require grad yields no parameter gradients, and CLIP's input
  comes from its frozen embeddings. So `--unfreeze_n_vision_tower_layers 12` trained
  nothing. `train.py` now selects non-reentrant checkpointing (numerically equivalent).
- **Gradient-accumulation normalization (silent).** transformers >= 4.46 skips the
  `/ gradient_accumulation_steps` division when the model's forward takes `**kwargs`
  (ours does), assuming the model normalizes itself. `compute_loss` returns a per-batch
  mean as in 4.31, so `LLaVATrainer` restores the original normalization; otherwise every
  gradient would be scaled by 4.
- **fp32 loss.** 4.31's Llama always returned fp32 logits; 4.57 keeps bf16. Logits are
  upcast before the token-weighted cross-entropy, as before.
- **Seeding.** The Trainer seeds only in its `__init__`, after the image pooler and LoRA
  weights were randomly initialized, so no two original runs started from the same weights.
  `train.py` seeds first, and `--data_seed` pins the length-grouped sampler's order.
- **API drift:** `load_in_4bit` only via `quantization_config`; new signatures of
  `_get_train_sampler`, `compute_loss`, `_save_checkpoint`.

### Dead code removed (~2780 lines)

- **The MPT model subtree.** MM2SG is Llama-based and never reaches it, and it imported
  symbols that no longer exist in transformers.
- **The three attention monkeypatches** (flash-attn, xformers, `llama_patch`) and
  `train_xformers.py`.
  - Attention is selected natively now.
  - On 4.57 the patches would still import cleanly while replacing `LlamaAttention.forward`
    with a 4.31-signature function, so they would fail *silently*.
  - The one non-attention helper they contained, `upcast_layer_for_flash_attention`, is
    preserved byte-identical in `llava/train/dtype_utils.py`.

### Compiled dependencies dropped

- **torch-scatter.** `multimodal_projector/compat.py` uses it if it is installed;
  otherwise it falls back to native ATen `segment_csr` in `native_ops.py`. The fallback is
  exact for max and within ~1e-7 for mean.
- **open3d.** Replaced by `llava/utils_pcd.py`, a dependency-free PCD reader. It is
  bit-exact against `open3d.io.read_point_cloud` on 12 released files, including the
  mm → m scaling.
- **flash-attn in PointTransformerV3.** If flash-attn is missing, PTv3 falls back to its
  non-flash path with a warning. `MMOR_REQUIRE_FLASH_ATTN=1` turns the fallback into an
  error. This matters because the non-flash path groups points differently from the
  trained model, so it is **not** numerically equivalent.

### Usability and robustness

- **Former hardcoded literals are CLI flags now.** Every default reproduces the old value,
  and the derived dataset filenames are verified byte-identical.
  - `main.py`: `--mode`, `--batch-size`, `--no-4dor`, `--limit-val-batches`.
  - `generate_dataset_format_for_llava.py`: `--n-perm`, `--add-temporal`, `--split`,
    `--drop-history`, `--no-4dor`, and others.
  - Same for `generate_downstream_dataset_format_for_llava.py`.
- **Weight caches no longer depend on the working directory.** They were addressed
  relative to the process CWD, so `_init_weights` reported "No weights found" when run
  from anywhere other than `LLaVA/`. They are now anchored to their module.
- **Generated artifacts are no longer tracked.** `__pycache__/*.pyc` were committed
  upstream, and importing the package dirtied 16 files. The PTv3 weight cache and the
  dataset symlink roots are gitignored too.

---

## 2. Dataset: deviations from the release and what has to be generated

The release does **not** contain everything the README says it does. These are the steps
and fixes needed to get from the published download to a dataset that trains and
evaluates.

### Layout

- `MM-OR_data` must point at the directory that **directly** contains the take folders
  (`001_PKA`, …, `take_jsons`). The official download script unzips one level deeper, into
  `MM-OR_data/MM-OR_processed/`, so the nested directory is the correct target.
- **4D-OR is effectively mandatory.** The published labels are *HybridOR*: about 30 % of
  each split comes from 4D-OR, and `ORDataset` defaults to `load_4dor=True`. Use
  `--no-4dor` for MM-OR-only runs, but those are not paper-comparable.
  - Scene graphs need only part of 4D-OR, per take: `colorimage/` plus
    `timestamp_to_pcd_and_frames_list.json` from `4D-OR.zip`, `pcds_sparse/` (shipped with
    the MM-OR release) and GT `segmentation_export_{1,2,5}/` (HF `egeozsoy/MM-OR`).
  - Depth, fused clouds, 3D boxes and 6D poses are not needed.
- An older 338 GB MM-OR export exists that lacks `relation_labels`, which is the
  scene-graph ground truth, along with `dino_features_l`, `panoptic_seg_*` and `pose`.
  It fails silently rather than loudly. Use the current release.

### Per-timepoint modality folders: generated, not shipped

The README says the preprocessed versions are included. They are not, so they have to be
generated with these scripts (run from `scene_graph_generation/`):

| folder | script |
|---|---|
| `take_audio_per_timepoint` | `create_take_sample_audios.py` |
| `take_audio_embeddings_per_timepoint` | `create_take_sample_audio_embeddings.py` |
| `take_transcripts_per_timepoint` | `create_take_sample_speech_transcripts.py` |
| `take_segmasks_per_timepoint` (MM-OR and 4D-OR) | `create_take_sample_segmasks.py` |

Fixes made to these scripts:

- **`create_take_sample_segmasks.py`** used bare `Path('MM-OR_data')` /
  `Path('4D-OR_data')` for two of its paths. That only resolves with CWD at the repo root,
  while its imports need CWD at `scene_graph_generation/`, so both cannot hold. It now
  uses the configured roots.
- **`create_take_sample_audios.py`** called `mkdir` without `parents=True`, so it failed
  on a fresh dataset where the parent folder does not exist.
- **Predicted segmasks (`PREDSEGMASKS`) do not need the panoptic-segmentation stack.**
  MM-OR ships the DVIS++/CTVIS predictions per take as `panoptic_seg_<cam>_for_val/`,
  already in the `TRACK_TO_METAINFO` palette. The script now uses them directly, with the
  `pan_pred` layout as fallback. The output is 32×32 uint8 label maps named
  `<ts>_<i>_GTFalse.png`, which is exactly what the dataloader expects with `USE_GT=false`.
- **Sharding and resuming.** Both the segmask and the audio-embedding scripts accept take
  names on argv.
  - Segmasks split work into (take, timestamp-shard) items (`SEGMASK_WORKERS`,
    `SEGMASK_SHARDS_PER_TAKE`) and skip timestamps that are already written.
  - Embeddings take `EMB_NUM_WORKERS` and skip undecodable clips with a warning instead of
    aborting.

### Known gaps in the release (affect both stacks equally)

- **`008_PKA/003512.mp3`** is a truncated 555-byte final clip and cannot be decoded. It is
  the only one: every other take has full audio-embedding coverage, and this one timestamp
  has no audio embedding.
- **`009_TKA`** has no segmentation source at all, so it has 0 segmasks, which is correct.
- **4D-OR has no predicted segmasks.** The checkpoint is a `PREDSEGMASKS` model, but
  4D-OR ships only GT masks. Predicted masks would require the CTVIS inference output,
  which is not released. The 4D-OR test takes (`002`, `006`) are therefore evaluated
  **without segmasks**. That is a deviation from the paper's setup, though identical for
  both stacks.
- **Missing modalities are skipped silently.** `or_dataset.py` adds audio/segmask inputs
  only `if path.exists()`, so any run started before preprocessing finishes is a run with
  those modalities dropped. Evaluate only on a complete, frozen dataset.
- **`extract_metalabels_for_MMOR.py` was never published.** It is imported by
  `downstream_prediction_model.py`, so the three downstream tasks (sterility breach, next
  action, robot phase) cannot run. Core scene-graph generation is unaffected.

---

## 3. Verification

The method throughout is to run the original environment (frozen pins, untouched
semantics) and the modernized one on the same code, data and hardware (NVIDIA A40,
sm_86), and compare outputs. Comparisons are made against a measured run-to-run noise
floor rather than bit-equality, because the sparse-conv and attention kernels accumulate
with atomics.

### Component level

| check | result |
|---|---|
| Module imports | 12/12 baseline, 15/15 modern |
| Model smoke test: forward, loss, cache, `generate()`, modality passthrough | 7/7 in both stacks; **identical loss 4.9313** |
| PointTransformerV3 outputs (feat / pooled / projected) | **cos = 1.000000**; every delta below the in-env noise floor |
| `spconv` `SubMConv3d` | bit-identical between stacks |
| native `segment_csr` vs torch-scatter | max: exact; mean: 6e-8 |
| PCD reader vs open3d | bit-exact on 12 released files |
| flash-attn 2.8.3 vs torch SDPA | max diff 2.4e-4 (fp16) |
| **PTv3 on Blackwell** (RTX PRO 6000, sm_120, PTX JIT) | **6/6, cos ≥ 0.999999** vs A40; `spconv` bit-identical to A40 (max_abs 3.3e-7 vs a dense conv3d reference on both) |

### End-to-end: full test set

Setup:
- Checkpoint `llava-v1.5-7b-task-lora_hybridor_qlora_20perm_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_0.50drop_MIXED`
  on `liuhaotian/llava-v1.5-7b`.
- Evaluated with `main.py --mode evaluate`, batch size 8, on one A40 per stack.
- Same frozen dataset for both. Modern run with `MMOR_REQUIRE_FLASH_ATTN=1`.
- Runtime: baseline 2 h 16 min, modern 2 h 12 min.

Relation Macro-F1:

| take | original stack | modernized | Δ |
|---|---|---|---|
| 002_4DOR | 0.907 | 0.908 | +0.001 |
| 006_4DOR | 0.880 | 0.887 | +0.007 |
| 004_PKA | 0.469 | 0.464 | −0.005 |
| 011_TKA | 0.502 | 0.498 | −0.004 |
| 036_PKA | 0.538 | 0.538 | 0.000 |
| 038_TKA | 0.516 | 0.513 | −0.003 |
| **4D-OR overall** | **0.895** | **0.899** | +0.004 |
| **MM-OR overall** | **0.501** | **0.500** | −0.001 |
| **combined** | **0.689** | **0.691** | +0.002 |

Other metrics on the combined test set:

| metric | original | modernized |
|---|---|---|
| Macro precision / recall | 0.800 / 0.635 | 0.803 / 0.636 |
| Binary interaction macro-F1 | 0.901 | 0.902 |
| Entity micro-F1 | 0.936 | 0.936 |

**Verdict: parity.** The deltas have no consistent sign and are the size expected from
greedy decoding resolving near-ties differently under different kernels (flash-attn 2.3.4
→ 2.8.3, a different bf16 GEMM stack). Inspecting raw generations on a padded batch
confirms this: 7 of 8 rows produced the same triplet set as the original, and the eighth
produced one extra triplet.

For reference, before the padding fix in §1 the same run scored MM-OR 0.106, while 4D-OR
was unchanged at 0.895. It also ran about 20 % slower, because the broken generations ran
to `max_new_tokens` instead of stopping at `</SG>`.

### Training: 200-step QLoRA runs

Setup: the README recipe (QLoRA r=128, bs 4 × GA 4, DeepSpeed ZeRO-2, bf16, gradient
checkpointing, image-order and modality-drop augmentation), one A40 per run, 200 optimizer
steps, identical batch order (`--data_seed 42`) in both stacks, two init seeds (42, 7).
The modern runs use `--max_grad_norm 0` (see the clipping note below).

Training loss, modern relative to original, mean per window:

| steps | seed 42 | seed 7 | original: seed 7 vs seed 42 |
|---|---|---|---|
| 11–50 | −0.4% | −0.5% | +0.4% |
| 51–100 | −0.3% | −0.5% | +0.4% |
| 101–150 | −0.5% | −0.3% | +0.6% |
| 151–200 | −0.4% | −0.4% | +1.3% |

The median per-step difference between stacks is 0.4–0.5%, against 1.0% between two seeds
of the original stack. **Verdict: parity.** The very first steps agree to 0.2–0.4%, and
Adam's state after step 1 shows matching gradients per parameter group (cosine 0.89–0.998).
The exception is LoRA B, because the two peft versions draw different random LoRA A
matrices; that is an init difference, not a computational one.

Before the CLIP-gradient fix, modern ran 2–5% *below* the original in every window, for
both seeds. That gap is how the bug was found.

Throughput on an A40: 14.8–16.4 s per step (modern) vs 17.7 s (original); peak GPU
memory 22 GB vs 40 GB. A full stage-1 epoch (312,300 samples, 19,518 steps) takes about
3.3 days on one A40.

**Gradient clipping never ran in the original training.** DeepSpeed 0.9.5's ZeRO-1/2
optimizer computes the gradient norm, and therefore clips, only for fp16
(`if self.dtype == torch.float16`). Under bf16, `--max_grad_norm 0.1` was a no-op, and
gradient norms of 2–4 went to Adam unclipped. DeepSpeed 0.19 does clip. The recipe flag
is unchanged (`train_sgg.sh` defaults to 0.1). To reproduce the original training
behaviour, pass `MAX_GRAD_NORM=0`. Clipping had no measurable effect on the 200-step
curves, which is expected: Adam is nearly invariant to a uniform gradient scale.

### Not yet verified

- **Comparison against the paper's published table.** This verification establishes
  *original pipeline == modernized pipeline*. It does not establish *either == paper*.
  The 4D-OR takes run without segmasks (see §2), so they deviate from the paper's setup.
- **Full training.** Only 200 steps (about 1% of a stage-1 epoch) were run per stack. A
  full stage-1 and stage-2 (temporal curriculum) training, followed by the test-set eval,
  is what would show whether the modern stack reproduces the released checkpoints.
- **Downstream tasks.** Blocked by the missing `extract_metalabels_for_MMOR.py`.
