import subprocess
import time

import modal

from vllm_autoserve import common

tag = "12.9.1-devel-ubuntu22.04"
vllm_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.12"
).uv_pip_install("vllm>=0.11.0", "requests")

vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

with vllm_image.imports():
    import requests


@common.app.cls(gpu="H200", image=vllm_image)
@modal.concurrent(target_inputs=20, max_inputs=100)
class VLLMServe:
    model_path: str = modal.parameter()

    @modal.enter()
    def up(self):
        vllm_cmd = [
            "vllm",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            f"{self.model_path}",
        ]
        self.vllm_process = subprocess.Popen(vllm_cmd)

        # Wait for server to be ready
        print("Waiting for vLLM server to be ready...")
        timeout = 600  # 10 minutes
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = requests.get("http://localhost:8000/health", timeout=1)
                if response.status_code == 200:
                    print("vLLM server is ready!")
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            raise RuntimeError(
                f"vLLM server did not become ready after {timeout} seconds"
            )

    @modal.web_server(port=8000)
    def serve(self):
        pass

    @modal.method()
    def boot(self):
        pass

    @modal.exit()
    def down(self):
        deadline = time.time() + 29.5  # 30s deadline
        while time.time() < deadline:
            try:
                response = requests.get("http://localhost:8000/load")
                if response.status_code == 200:
                    load_metrics = response.json()
                    server_load = load_metrics.get("server_load")
                    print(f"Server load: {server_load} requests")
                    if server_load is None:
                        raise RuntimeError(
                            f"Server load expected from /load response, but found None: {load_metrics}"
                        )
                    if server_load == 0:
                        print("Server load is 0, continuing...")
                        break
                else:
                    print(f"Failed to get load metrics: {response.status_code}")
            except Exception as e:
                print(f"Error getting load metrics: {e}")

            time.sleep(0.5)  # Wait half-second before next check
        else:
            print("Deadline reached, continuing regardless of server load...")

        self.vllm_process.terminate()
