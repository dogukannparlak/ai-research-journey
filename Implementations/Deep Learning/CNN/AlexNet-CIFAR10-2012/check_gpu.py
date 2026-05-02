"""
GPU and CUDA availability check for AlexNet-CIFAR10-2012.
Run before training to verify the environment is correctly configured.
"""


def check_gpu() -> str:
    print("GPU / CUDA Availability Check")
    print("=" * 50)

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed.")
        print("Run: pip install -r requirements.txt")
        return "cpu"

    print(f"PyTorch version : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available  : {cuda_available}")

    if not cuda_available:
        print("\nPossible reasons:")
        print("  - CPU-only PyTorch is installed (most likely)")
        print("  - NVIDIA drivers not installed or outdated")
        print("  - No NVIDIA GPU in system")
        print("\nTo install CUDA-enabled PyTorch:")
        print("  pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \\")
        print("      --index-url https://download.pytorch.org/whl/cu124")
        return "cpu"

    device_count = torch.cuda.device_count()
    print(f"GPU count       : {device_count}")
    print()

    for i in range(device_count):
        props      = torch.cuda.get_device_properties(i)
        name       = props.name
        memory_gb  = props.total_memory / 1024 ** 3
        compute    = f"{props.major}.{props.minor}"
        print(f"  GPU {i}  : {name}")
        print(f"  VRAM    : {memory_gb:.1f} GB")
        print(f"  Compute : {compute}")

    print()
    print(f"cuDNN available : {torch.backends.cudnn.is_available()}")
    print(f"cuDNN version   : {torch.backends.cudnn.version()}")

    print("\nRunning CUDA functional test...")
    try:
        device      = torch.device("cuda:0")
        a           = torch.rand(2048, 2048, device=device)
        b           = torch.rand(2048, 2048, device=device)
        _           = torch.matmul(a, b)
        torch.cuda.synchronize()
        allocated   = torch.cuda.memory_allocated() / 1024 ** 2
        reserved    = torch.cuda.memory_reserved()  / 1024 ** 2
        print(f"  Matrix multiply (2048×2048) : OK")
        print(f"  Memory allocated            : {allocated:.1f} MB")
        print(f"  Memory reserved             : {reserved:.1f} MB")
        del a, b
        torch.cuda.empty_cache()
        print("  CUDA functional test        : PASSED")
    except Exception as e:
        print(f"  CUDA functional test        : FAILED — {e}")
        return "cpu"

    print()
    print("Recommended device: cuda")
    print("config.py is already set to DEVICE = \"cuda\"")
    return "cuda"


if __name__ == "__main__":
    check_gpu()
