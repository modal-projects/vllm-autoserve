import asyncio
import json
import os

import modal

from vllm_autoserve import common

gateway_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi", "huggingface_hub", "httpx")
    .add_local_python_source("vllm_autoserve")
)
vllm_gateway_auth = modal.Secret.from_name(
    "vllm-gateway-auth", required_keys=["VLLM_GATEWAY_AUTH"]
)

# Hop-by-hop headers that should not be forwarded (RFC 2616)
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


with gateway_image.imports():
    import httpx
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import Response, StreamingResponse
    from huggingface_hub import model_info
    from pydantic import BaseModel
    from starlette.background import BackgroundTask


@common.app.cls(
    image=gateway_image,
    volumes={"/root/.cache/huggingface": common.hf_cache},
    secrets=[vllm_gateway_auth, common.hf_secret],
    min_containers=1,
)
@modal.concurrent(max_inputs=800)
class Gateway:
    @modal.enter()
    def up(self):
        expected_token = os.environ.get("VLLM_GATEWAY_AUTH")
        if expected_token is None:
            raise RuntimeError("Can't find valid VLLM_GATEWAY_AUTH secret.")
        self.expected_token = expected_token

    @modal.asgi_app(label="gateway")
    def vllm_gateway(self):
        web_app = FastAPI()

        class GatewayRequest(BaseModel):
            model_path: str

        @web_app.post("/up")
        async def up(request: GatewayRequest, authorization: str = Header(None)):
            # auth
            self._auth(authorization)

            # standardize model for vllm server pool
            sanitized_model_path = _sanitize_model_path(request.model_path)

            # look up server pool for model
            VLLM = modal.Cls.from_name(common.app.name, "VLLMServe")
            vllm_pool = VLLM(model_path=sanitized_model_path)

            # start booting up a server in the pool before returning
            boot_call = await vllm_pool.boot.spawn.aio()
            try:
                boot_call.get(timeout=0.4)
            # still booting
            except asyncio.TimeoutError:
                return {
                    "status": "pending",
                    "model": sanitized_model_path,
                }
            # server is healthy
            else:
                return {
                    "status": "healthy",
                    "model": sanitized_model_path,
                }

        @web_app.post("/v1/chat/completions")
        async def proxy_chat_completions(
            request: Request, authorization: str = Header(None)
        ):
            # auth
            self._auth(authorization)

            # read model param from chat request + sanitize as pool identifier
            raw = await request.body()
            try:
                payload = json.loads(raw)
            except Exception:
                raise HTTPException(status_code=400, detail="Body must be valid JSON")
            model_value = payload.get("model")
            if not model_value:
                raise HTTPException(
                    status_code=400, detail="Request must include 'model'"
                )
            model_id = _sanitize_model_path(model_value)

            # look up server pool for model
            VLLM = modal.Cls.from_name(common.app.name, "VLLMServe")
            vllm_pool = VLLM(model_path=model_id)

            # block until pool is healthy
            await vllm_pool.boot.remote.aio()

            #
            base = vllm_pool.serve.get_web_url().rstrip(
                "/"
            )  # documented way to get the web URL
            target = f"{base}/v1/chat/completions"

            # Forward headers, excluding hop-by-hop and ones the client lib should set
            fwd_headers = {
                k: v
                for k, v in request.headers.items()
                if k.lower() not in HOP_BY_HOP_HEADERS | {"host", "content-length"}
            }

            is_stream = bool(payload.get("stream", False))

            if not is_stream:
                async with httpx.AsyncClient(timeout=None) as client:
                    resp = await client.post(
                        target,
                        headers=fwd_headers,
                        content=raw,
                        params={"model_path": model_id},
                    )
                    body = await resp.aread()  # bytes; pass-through unchanged

                return Response(
                    content=body,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"),
                    headers=_filtered_resp_headers(resp.headers),
                )

            # stream upstream -> downstream unchanged
            client = httpx.AsyncClient(timeout=None)
            req = client.build_request(
                "POST",
                target,
                headers=fwd_headers,
                content=raw,
                params={"model_path": model_id},
            )
            resp = await client.send(req, stream=True)

            media_type = resp.headers.get("content-type")

            async def _cleanup():
                await resp.aclose()
                await client.aclose()

            return StreamingResponse(
                resp.aiter_raw(),  # byte-for-byte pass-through
                status_code=resp.status_code,
                headers=_filtered_resp_headers(resp.headers),
                media_type=media_type,
                background=BackgroundTask(_cleanup),
            )

        @web_app.get("/health")
        async def health():
            return {"status": "ok"}

        return web_app

    def _auth(self, authorization):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Missing or invalid Authorization header"
            )

        token = authorization.replace("Bearer ", "", 1)
        if token != self.expected_token:
            raise HTTPException(status_code=403, detail="Invalid authentication token")


def _sanitize_model_path(model_path: str) -> str:
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


def _filtered_resp_headers(hdrs: dict[str, str]) -> dict[str, str]:
    """Filter response headers, removing hop-by-hop, content-length, and Modal-internal headers."""
    block = {"content-length"}

    def keep(k: str) -> bool:
        kl = k.lower()
        if kl in HOP_BY_HOP_HEADERS or kl in block:
            return False
        if kl.startswith("modal-"):  # strip Modal-internal headers
            return False
        return True

    return {k: v for k, v in hdrs.items() if keep(k)}


# Make it deployable
app = common.app
