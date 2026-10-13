# Entrypoint for training. Kept for compatibility with the documented command
# `python -m llava.train.train_mem`.
#
# This file used to call replace_llama_attn_with_flash_attn() before importing
# transformers, to swap in a hand-written FlashAttention forward. That is obsolete and
# actively harmful on modern transformers: LlamaModel._prepare_decoder_attention_mask no
# longer exists, while apply_rotary_pos_emb/repeat_kv still import fine, so the patch
# would load cleanly and then replace LlamaAttention.forward with an old-signature
# function -- failing silently rather than loudly.
#
# Attention is now selected natively via `attn_implementation` ("sdpa" by default,
# "flash_attention_2" when available); see llava/train/train.py.
from llava.train.train import train

if __name__ == "__main__":
    train()
