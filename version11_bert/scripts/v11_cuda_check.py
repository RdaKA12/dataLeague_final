"""Validate CUDA readiness for V11 training."""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"cuda_ready": False, "error": f"torch import failed: {exc}"}, indent=2))
        return 1

    payload = {
        "torch_version": torch.__version__,
        "cuda_ready": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["cuda_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

