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
set (combined Macro-F1 0.691 vs 0.689, every take within ±0.007; see
[Verification](#verification)).**

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

### Not yet verified

- **Comparison against the paper's published table.** This verification establishes
  *original pipeline == modernized pipeline*. It does not establish *either == paper*.
  The 4D-OR takes run without segmasks (see §2), so they deviate from the paper's setup.
- **Training.** Only the evaluation path has been exercised end-to-end. Training was
  checked at the level of imports and the smoke test only.
- **Downstream tasks.** Blocked by the missing `extract_metalabels_for_MMOR.py`.
