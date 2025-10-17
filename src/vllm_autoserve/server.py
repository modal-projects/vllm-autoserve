import subprocess
import time

import modal
import modal.experimental

from vllm_autoserve import common
from vllm_autoserve.hf_utils import get_base_model_from_model_card

tag = "12.9.1-devel-ubuntu22.04"
vllm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm>=0.11.0", "requests", "flashinfer-python")
)


with vllm_image.imports():
    import requests

@common.app.cls(
    gpu="H200:2",
    image=vllm_image,
    timeout=30 * 60,  # 1 hour, for downloads
    scaledown_window=10 * 60,  # 15 minutes
    secrets=[common.hf_secret, common.vllm_gateway_auth],
    volumes={
        "/root/.cache/huggingface": common.hf_cache,
        "/merged_models": common.merged_models,
    }
)
@modal.concurrent(target_inputs=20, max_inputs=100)
class VLLMServe:
    model_path: str = modal.parameter()

    @modal.enter()
    def up(self):
        import threading
        import re
        import os

        print("Checking if model is PEFT adapter or full model...")
        check_peft = modal.Function.from_name(common.app.name, common.INSPECT_HF_REPO_FUNC_NAME)
        hf_token = os.environ.get("HF_TOKEN")
        expected_auth_token = os.environ.get("VLLM_GATEWAY_AUTH")
        model_path_to_load = str(self.model_path)
        peft_info = check_peft.remote(self.model_path, token=hf_token)

        print(f"PEFT inspection result: {peft_info}")
        if not peft_info.is_full_model:
            print(f"Model at {self.model_path} does not appear to be a full model: {peft_info}")
            print(f"Attempting to merge PEFT adapter into base model...")
            merge_peft = modal.Function.from_name(common.app.name, common.MERGE_PEFT_FUNC_NAME)
            base_model_to_use = peft_info.base_model_name_or_path

            # hacky check for MLX models
            if peft_info.base_model_name_or_path.startswith("mlx"):
                print("Detected MLX base model, get original model name from model card...")
                base_model_from_mlx = get_base_model_from_model_card(
                    peft_info.base_model_name_or_path, token=hf_token
                )
                if base_model_from_mlx is not None:
                    print(f"Found base model from model card: {base_model_from_mlx}")
                    base_model_to_use = base_model_from_mlx

            # TODO: remove once ToS issues are resolved
            # hacky check for google models, test with unsloth version of the model til ToS issues are resolved
            if base_model_to_use.startswith("google"):
                base_model_to_use = base_model_to_use.replace("google/", "unsloth/")
            model_path_to_load = merge_peft.remote(base_model_to_use, self.model_path, token=hf_token)

        print(subprocess.run(
            ["ls", "-lah", "/root/.cache/huggingface"],
            capture_output=True,
            text=True,
        ).stdout)
        print(subprocess.run(
            ["ls", "-lah", "/root/.cache/huggingface/hub"],
            capture_output=True,
            text=True,
        ).stdout)
        print(subprocess.run(
            ["ls", "-lah", "/merged_models"],
            capture_output=True,
            text=True,
        ).stdout)
        print(f"Starting vLLM server with model at: {model_path_to_load}")

        vllm_cmd = [
            "vllm",
            "serve",
            model_path_to_load,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--tensor-parallel-size",
            "2",
            "--enforce-eager",
            "--api-key",
            expected_auth_token,
        ]
        self.vllm_error = None
        self.vllm_process = subprocess.Popen(
            vllm_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Pattern to detect fatal errors in logs
        fatal_re = re.compile(
            r"(Traceback \(most recent call last\)|RuntimeError:|ValidationError:|pydantic_core\._pydantic_core\.ValidationError)"
        )
        saw_fatal_pre_health = False
        healthy = False
        log_lines = []

        def tail():
            nonlocal saw_fatal_pre_health
            for line in self.vllm_process.stdout:
                print(line, end="")  # mirror logs
                log_lines.append(line)
                if not healthy and fatal_re.search(line):
                    saw_fatal_pre_health = True

        t = threading.Thread(target=tail, daemon=True)
        t.start()

        # Wait for server to be ready
        print("Waiting for vLLM server to be ready...")
        timeout = 30 * 60  # 30 minutes
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check if vllm_process has terminated
            poll_result = self.vllm_process.poll()
            if poll_result is not None:
                # Process has terminated
                error_msg = f"vLLM process terminated with exit code {poll_result}.\n"
                error_msg += "Full logs:\n" + "".join(log_lines)
                self.vllm_error = error_msg
                return

            if saw_fatal_pre_health:
                # Fatal error detected before health check passed
                error_msg = (
                    "Fatal error detected in logs before server became healthy.\n"
                )
                error_msg += "Full logs:\n" + "".join(log_lines)
                self.vllm_error = error_msg
                self.vllm_process.terminate()
                subprocess.Popen(
                    ["python", "-m", "http.server", "8000"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    self.vllm_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.vllm_process.kill()
                return

            try:
                response = requests.get("http://localhost:8000/health", timeout=1)
                if response.status_code == 200:
                    healthy = True
                    print("vLLM server is ready!")
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            error_msg = f"vLLM server did not become ready after {timeout} seconds.\n"
            error_msg += "Full logs:\n" + "".join(log_lines)
            self.vllm_error = error_msg
            self.vllm_process.terminate()
            return

    @modal.web_server(
        port=8000, startup_timeout=35 * 60
    )  # >5min to allow for retrieving error via boot
    def serve(self):
        pass

    @modal.method()
    def boot(self):
        if self.vllm_error is not None:
            modal.experimental.stop_fetching_inputs()
            return self.vllm_error
        return

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
