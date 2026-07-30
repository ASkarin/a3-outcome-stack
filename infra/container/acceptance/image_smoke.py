from __future__ import annotations

import importlib
import importlib.metadata
import json

import torch
import torchvision


def main() -> int:
    modules = (
        "lerobot",
        "lerobot.policies.smolvla.modeling_smolvla",
        "tensorboard",
        "torchcodec",
        "wandb",
    )
    for module in modules:
        importlib.import_module(module)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 3:
        raise RuntimeError(f"expected three GPUs, found {torch.cuda.device_count()}")

    devices = []
    for index in range(torch.cuda.device_count()):
        tensor = torch.ones(1024, device=f"cuda:{index}")
        torch.cuda.synchronize(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "sum": tensor.sum().item(),
            }
        )

    payload = {
        "schema_version": 1,
        "status": "pass",
        "python_packages": {
            "lerobot": importlib.metadata.version("lerobot"),
            "tensorboard": importlib.metadata.version("tensorboard"),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "cuda_runtime": torch.version.cuda,
        "devices": devices,
        "imports": list(modules),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
