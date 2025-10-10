vLLM Autoserve
=========

## Setup: Modal Secrets
- Create gateway secret: `modal secret vllm-gateway-auth VLLM_GATEWAY_AUTH=...`
  - Hint: use `secrets.token_urlsafe(n_bytes)` to generate.
- Create huggingface secret: `modal secret autoserve-hf-secret HF_TOKEN=...` or through the Modal dashboard.

## Install this package
```
uv sync
```

## Deploy
```
modal deploy -m vllm_autoserve.gateway
```

## Usage
Assuming the `vllm-gateway-auth` secret was saved at `.vllm-gateway-auth`, spin up a new container pool for a model:
```console
$ curl -LX POST https://{WORKSPACE}-{ENVIRONMENT}--gateway.modal.run/up -H "Authorization: Bearer $(cat .vllm-gateway-auth)" -H "Content-Type: application/json" -d '{"model_path": "Qwen/Qwen3-Next-80B-A3B-Instruct"}'
{"status":"pending","model":"Qwen/Qwen3-Next-80B-A3B-Instruct"}

$ curl -LX POST https://{WORKSPACE}-{ENVIRONMENT}--gateway.modal.run/up -H "Authorization: Bearer $(cat .vllm-gateway-auth)" -H "Content-Type: application/json" -d '{"model_path": "Qwen/Qwen3-Next-80B-A3B-Instruct"}'
{"status":"healthy","model":"Qwen/Qwen3-Next-80B-A3B-Instruct"}
```
Then, request chat completions directly from the gateway:
```console
$ curl -LX POST https://{WORKSPACE}-{ENVIRONMENT}--gateway.modal.run/v1/chat/completions -H "Authorization: Bearer $(cat .vllm-gateway-auth)" -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-Next-80B-A3B-Instruct", "messages": [{"role": "user", "content": "Hello, how are you?"}]}'
{"id":"chatcmpl-da25fbde39eb48d981b07670c3ce3d82","object":"chat.completion","created":1760073333,"model":"Qwen/Qwen3-Next-80B-A3B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! I'm doing great, thank you for asking! 😊 How about you? I hope you're having a wonderful day! Let me know if there's anything I can help you with—I'm here to chat or assist with any questions! 🌟","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning_content":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":14,"total_tokens":68,"completion_tokens":54,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

Or, with `stream=True`:
```console
$ curl -LX POST https://{WORKSPACE}-{ENVIRONMENT}--gateway.modal.run/v1/chat/completions -H "Authorization: Bearer $(cat .vllm-gateway-auth)" -H "Content-Type: application/json" -d '{"model": "Qwen/Qwen3-Next-80B-A3B-Instruct", "stream": true, "messages": [{"role": "user", "content": "Hello, how are you?"}]}'
data: {"id":"chatcmpl-edcac873c27143ddad0085ebab042a40","object":"chat.completion.chunk","created":1760073516,"model":"Qwen/Qwen3-Next-80B-A3B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}],"prompt_token_ids":null}

data: {"id":"chatcmpl-edcac873c27143ddad0085ebab042a40","object":"chat.completion.chunk","created":1760073516,"model":"Qwen/Qwen3-Next-80B-A3B-Instruct","choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-edcac873c27143ddad0085ebab042a40","object":"chat.completion.chunk","created":1760073516,"model":"Qwen/Qwen3-Next-80B-A3B-Instruct","choices":[{"index":0,"delta":{"content":"!"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

data: {"id":"chatcmpl-edcac873c27143ddad0085ebab042a40","object":"chat.completion.chunk","created":1760073516,"model":"Qwen/Qwen3-Next-80B-A3B-Instruct","choices":[{"index":0,"delta":{"content":" I"},"logprobs":null,"finish_reason":null,"token_ids":null}]}

...
```
