import os
import modal
from dataclasses import dataclass

app = modal.App("vllm-autoserve")
hf_utils_app = modal.App("vllm-autoserve-hf-utils")
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
hf_secret_name = os.environ.get("VLLM_AUTOSERVE_SECRET", "autoserve-hf-secret")
hf_secret = modal.Secret.from_name(hf_secret_name)

INSPECT_HF_REPO_FUNC_NAME = "inspect_hf_repo_for_peft"
MERGE_PEFT_FUNC_NAME = "merge_model_and_save_to_volume"

@dataclass
class PeftInfo:
    is_full_model: bool = False
    is_peft_adapter: bool = False
    base_model_name_or_path: str | None = None
    detected_from: str | None = None
    adapter_config: dict | None = None