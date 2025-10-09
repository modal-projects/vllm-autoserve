import modal

app = modal.App("vllm-autoserve")
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
