import modal

from vllm_autoserve import common
from vllm_autoserve.common import PeftInfo

hf_image = (
    modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install("huggingface_hub")
)

app = common.app

def get_base_model_from_model_card(model_repo: str, token=None) -> str | None:
    from huggingface_hub import model_info
    try:
        meta = model_info(model_repo, token=token)
        if meta.cardData and "base_model" in meta.cardData:
            return meta.cardData["base_model"]
        else:
            return None
    except Exception as e:
        print(f"Failed to get model info for {model_repo}: {e}")
        return None

@app.function(
    image=hf_image.add_local_python_source("vllm_autoserve"),
    volumes={
        "/root/.cache/huggingface": common.hf_cache,
    },
)
def inspect_hf_repo_for_peft(repo_id: str, revision=None, token=None):
    from huggingface_hub import list_repo_files, hf_hub_download, model_info
    import json
    print(f"Inspecting repo {repo_id} at revision {revision} for PEFT adapter info...")

    info = {}
    try:
        files = list_repo_files(repo_id, revision=revision, token=token)
        info["found_files"] = files
    except Exception as e:
        return {"error": f"Failed to list files: {e}"}

    result = {
        "is_full_model": False,
        "is_peft_adapter": False,
        "base_model_name_or_path": None,
        "detected_from": None,
        "adapter_config": None,
    }

    # --- 1️⃣ Try adapter config ---
    for candidate in ["adapter_config.json", "peft_config.json"]:
        if candidate in files:
            path = hf_hub_download(repo_id, candidate, revision=revision, token=token)
            print(path)
            cfg = json.load(open(path))
            print(cfg)
            result["adapter_config"] = cfg
            for key in ["base_model_name_or_path", "pretrained_model_name_or_path", "base_model"]:
                if key in cfg:
                    result["base_model_name_or_path"] = cfg[key]
                    result["detected_from"] = candidate
                    break
            result["is_peft_adapter"] = True
            break

    # --- 2️⃣ Try repo metadata (Hugging Face model card info) ---
    base_model_from_card = get_base_model_from_model_card(model_repo=repo_id, token=token)
    if base_model_from_card is not None:
        result["base_model_name_or_path"] = base_model_from_card
        result["is_peft_adapter"] = True
        result["detected_from"] = "model_card"


    # --- 3️⃣ Heuristic: if adapter_model.bin present and no config.json, likely PEFT adapter ---
    if any(f in files for f in ["adapter_model.bin", "adapter_model.safetensors"]) and not any(f == "config.json" for f in files):
        result["is_peft_adapter"] = True

    # --- 4️⃣ Determine full model ---
    if "config.json" in files and "tokenizer_config.json" in files:
        result["is_full_model"] = True

    # validate there is only a single base model listed
    if result["base_model_name_or_path"] is not None and isinstance(result["base_model_name_or_path"], list):
        # check length is 1
        if len(result["base_model_name_or_path"]) != 1:
            raise ValueError("Multiple base models found in adapter config, cannot determine single base model.")
        result["base_model_name_or_path"] = result["base_model_name_or_path"][0]

    return PeftInfo(**result)


@app.function(
    image=hf_image.uv_pip_install(["transformers", "peft", "torch"]).add_local_python_source("vllm_autoserve"),
    volumes={
        "/root/.cache/huggingface": common.hf_cache,
    },
)
def merge_model_and_save_to_volume(base_model_repo: str, peft_repo: str, revision=None, token=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    print(f"Merging PEFT adapter from {peft_repo} into base model {base_model_repo}...")

    merged_model_name = peft_repo + "-merged"
    hf_subdir = peft_repo.replace("/", "--")
    # hf cache structure
    hf_cache_path = f"/root/.cache/huggingface/hub/models--{merged_model_name}"

    # 1) load base model (use device_map and dtype for large models)
    print("Loading base model...")
    supported_model_loaders = [AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration]

    base = None
    for loader in supported_model_loaders:
        try:
            base = loader.from_pretrained(
                base_model_repo,
                device_map="auto",
                token=token,
            )
            print(f"Loaded base model using {loader.__name__}")
            break
        except Exception as e:
            print(f"Failed to load base model with {loader.__name__}: {e}")
    if base is None:
        raise ValueError(f"Failed to load base model {base_model_repo} with supported loaders.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_repo, token=token)

    # 2) load the adapter on top
    print("Loading PEFT adapter...")
    model_with_adapter = PeftModel.from_pretrained(base, peft_repo, device_map="auto", token=token)

    # 3) merge adapter weights into the base model
    # merge_and_unload() returns a transformers model (adapter weights applied)
    print("Merging adapter into base model...")
    merged_model = model_with_adapter.merge_and_unload()

    # 4) save merged model + tokenizer
    print(f"Saving merged model to {hf_cache_path} ...")
    merged_model.save_pretrained(hf_cache_path)
    print("Saving tokenizer...")
    tokenizer.save_pretrained(hf_cache_path)
    return hf_cache_path