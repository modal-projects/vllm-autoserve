import os
import modal

app = modal.App("vllm-autoserve")
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
hf_secret_name = os.environ.get("VLLM_AUTOSERVE_SECRET", "autoserve-hf-secret")
hf_secret = modal.Secret.from_name(hf_secret_name)
