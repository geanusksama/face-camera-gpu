import subprocess
import sys

GPU_PROVIDERS = ('CUDAExecutionProvider', 'DmlExecutionProvider')


def check_gpu() -> str:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    active = [p for p in GPU_PROVIDERS if p in providers]
    if not active:
        print("GPU/CUDA não disponível. Reconhecimento não iniciado.")
        sys.exit(1)

    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    gpu_name = result.stdout.strip() if result.returncode == 0 else "GPU"
    provider_tag = "CUDA" if 'CUDAExecutionProvider' in active else "DirectML"
    gpu_name = f"{gpu_name} [{provider_tag}]"
    print(f"GPU disponível: {gpu_name}")
    return gpu_name


def get_best_providers() -> list[str]:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'DmlExecutionProvider' in providers:
        return ['DmlExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']
