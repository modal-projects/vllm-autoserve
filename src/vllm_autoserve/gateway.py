from asyncio.events import Handle
import os
import modal

from vllm_autoserve import common
from vllm_autoserve import server

gateway_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi",
)
vllm_gateway_auth = modal.Secret.from_name(
    "vllm-gateway-auth", required_keys=["VLLM_GATEWAY_AUTH"]
)

with gateway_image.imports():
    from fastapi import Header, HTTPException
    from pydantic import BaseModel

    class GatewayRequest(BaseModel):
        model_path: str


@common.app.cls(
    image=gateway_image,
    volumes={"/root/.cache/huggingface": common.hf_cache},
    secrets=[vllm_gateway_auth],
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
        # Check Bearer token
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Missing or invalid Authorization header"
            )

        token = authorization.replace("Bearer ", "", 1)
        if token != self.expected_token:
            raise HTTPException(status_code=403, detail="Invalid authentication token")

        # TODO sanitize model_path
        try:
            cls = modal.Cls.from_name(common.app.name, "VLLMServe")
            vllm_url = cls(model_path=request.model_path).serve.get_web_url()
        except modal.exception.NotFoundError:
            # block until server is live
            vllm_url = server.VLLMServe(
                model_path=request.model_path
            ).serve.get_web_url()

        print(f"Model path: {request.model_path}")
        print(f"Inference")

        return {
            "status": "ok",
            "model_path": request.model_path,
            "inference_server_url": vllm_url,
        }


app = common.app
