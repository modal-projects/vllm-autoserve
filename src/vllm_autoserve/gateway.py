import os

import modal
from fastapi import Header, HTTPException
from pydantic import BaseModel

from vllm_autoserve import common

gateway_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi", "huggingface_hub")
    .add_local_python_source("vllm_autoserve")
)
vllm_gateway_auth = modal.Secret.from_name(
    "vllm-gateway-auth", required_keys=["VLLM_GATEWAY_AUTH"]
)


with gateway_image.imports():
    from huggingface_hub import model_info


class GatewayRequest(BaseModel):
    model_path: str


@common.app.cls(
    image=gateway_image,
    volumes={"/root/.cache/huggingface": common.hf_cache},
    secrets=[vllm_gateway_auth, common.hf_secret],
    min_containers=1,
)
@modal.concurrent(max_inputs=800)
class VllmGateway:
    @modal.enter()
    def up(self):
        expected_token = os.environ.get("VLLM_GATEWAY_AUTH")
        if expected_token is None:
            raise RuntimeError("Can't find valid VLLM_GATEWAY_AUTH secret.")
        self.expected_token = expected_token

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def vllm_gateway(
        self, request: GatewayRequest, authorization: str = Header(None)
    ):
        self._auth(authorization)

        sanitized_model_path = self._sanitize_model_path(request.model_path)

        cls = modal.Cls.from_name(common.app.name, "VLLMServe")
        requested_func = cls(model_path=sanitized_model_path)
        # boot up a vllm server before returning
        boot_call = requested_func.boot.spawn()
        result = boot_call.get(timeout=0.4)
        if result == "booted":
            vllm_url = requested_func.serve.get_web_url()
            return {
                "status": "healthy",
                "model": sanitized_model_path,
                "inference_server_url": vllm_url,
            }
        else:
            return {
                "status": "pending",
                "model": sanitized_model_path,
            }

    def _auth(self, authorization):
        # Check Bearer token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Missing or invalid Authorization header"
            )

        token = authorization.replace("Bearer ", "", 1)
        if token != self.expected_token:
            raise HTTPException(status_code=403, detail="Invalid authentication token")

    def _sanitize_model_path(self, model_path: str) -> str:
        """
        Sanitize model_path to extract the HuggingFace repo identifier.
        Handles both repo IDs (e.g., 'org/model') and full URLs (e.g., 'https://huggingface.co/org/model').
        """

        # Remove trailing slashes
        model_path = model_path.rstrip("/")

        # Check if it's a URL and extract the repo ID
        if model_path.startswith("http://") or model_path.startswith("https://"):
            # Parse the URL to extract repo identifier
            # Expected format: https://huggingface.co/org/model
            parts = model_path.split("huggingface.co/")
            if len(parts) == 2:
                repo_id = parts[1]
            else:
                raise HTTPException(
                    status_code=400, detail="Invalid HuggingFace URL format"
                )
        else:
            # Assume it's already a repo identifier
            repo_id = model_path

        # Validate the repo identifier format (should be org/model)
        if "/" not in repo_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid model path format. Expected 'org/model' or HuggingFace URL",
            )

        # Verify the model exists on HuggingFace
        try:
            model_info(repo_id)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found on HuggingFace: {repo_id}. Error: {str(e)}",
            )

        return repo_id


app = common.app
