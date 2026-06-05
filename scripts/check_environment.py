import importlib
import sys

PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "sklearn",
    "pydicom",
    "nibabel",
    "monai",
    "einops",
    "kornia",
    "natsort",
]


def main():
    print(f"Python: {sys.version.split()[0]}")
    failed = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {package}: {version}")
        except Exception as exc:
            failed.append((package, exc))
            print(f"[FAIL] {package}: {exc}")

    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    if failed:
        raise SystemExit("Some packages are missing. Please check requirements.txt.")


if __name__ == "__main__":
    main()
