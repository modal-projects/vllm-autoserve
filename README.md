vLLM Autoserve
=========

## Prerequisites
- Create gateway secret: `modal secret vllm-gateway-auth VLLM_GATEWAY_AUTH=...`
- Create huggingface secret: `modal secret autoserve-hf-secret HF_TOKEN=...` or through the dashboard

## Install
```
uv sync
```

## Deploy
```
modal deploy -m vllm_autoserve.gateway
```
