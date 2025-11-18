import torch
import platform
import os

def test_torch_cuda():
    print("===== PyTorch Environment Check =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDNN version: {torch.backends.cudnn.version()}")
    print(f"Device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"[GPU {i}] {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"  Memory Cached:    {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")

    print("\n===== System Information =====")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {platform.python_version()}")
    print(f"Current working directory: {os.getcwd()}")

    print("\n===== Simple CUDA Test =====")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create random tensors
    a = torch.randn((1000, 1000), device=device)
    b = torch.randn((1000, 1000), device=device)
    c = a + b

    # Verify results
    print(f"Computation on device: {device}")
    print(f"Sum tensor mean value: {c.mean().item():.4f}")

if __name__ == "__main__":
    test_torch_cuda()
