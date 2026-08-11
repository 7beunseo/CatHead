from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    """사용 가능한 장치를 점검해 안전한 실행 장치를 반환한다."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    """실험 재현성을 위해 Python / NumPy / PyTorch 시드를 동시에 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_torch_performance_flags(device: torch.device) -> None:
    """PyTorch에서 일반적으로 권장되는 성능 플래그를 켠다."""
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def _make_json_safe(payload: Any) -> Any:
    if is_dataclass(payload):
        payload = asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {key: _make_json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_make_json_safe(value) for value in payload]
    return payload


def save_json(path: Path, payload: Any) -> None:
    """딕셔너리/리스트/데이터클래스를 사람이 읽기 쉬운 JSON으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_make_json_safe(payload), file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def append_text_line(path: Path, text: str) -> None:
    """로그 파일에 한 줄씩 append 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(text.rstrip() + "\n")


def atomic_torch_save(path: Path, payload: Any) -> None:
    """중간 저장 실패로 깨진 체크포인트가 남지 않도록 임시 파일 후 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(5):
        # Use a unique temp file so concurrent/resumed runs cannot steal each other's
        # checkpoint before the atomic replace on Windows.
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            return
        except OSError as error:
            last_error = error
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            time.sleep(0.2 * (attempt + 1))
    raise last_error if last_error is not None else RuntimeError(f"Failed to save checkpoint: {path}")


def make_versioned_dir(base_dir: Path) -> Path:
    """
    같은 실험 이름이 여러 번 실행될 때 기존 결과를 덮어쓰지 않도록
    v1, v2, v3 ... 형태의 폴더를 자동 생성한다.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        candidate = base_dir / f"v{version}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        version += 1


def capture_rng_state() -> dict[str, Any]:
    """정확한 resume 을 위해 난수 생성기 상태까지 저장한다."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """저장된 난수 상태를 복원해 dropout / 셔플 흐름까지 이어간다."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.detach().to(device="cpu", dtype=torch.uint8)
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = []
        for cuda_state in state["torch_cuda"]:
            if isinstance(cuda_state, torch.Tensor):
                cuda_state = cuda_state.detach().to(device="cpu", dtype=torch.uint8)
            cuda_states.append(cuda_state)
        torch.cuda.set_rng_state_all(cuda_states)


def format_float(value: float | None) -> str:
    if value is None:
        return "None"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return f"{value}"
    return f"{value:.6f}"


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """마스크가 1인 위치만 평균을 내는 공통 유틸리티."""
    mask = mask.float()
    denom = mask.sum().clamp_min(eps)
    return (values * mask).sum() / denom
