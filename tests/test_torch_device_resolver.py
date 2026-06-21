"""Tests for ``statspai.utils._torch_device.resolve_torch_device``.

These tests cover the decision logic without requiring a CUDA box: the
``cuda`` branch is exercised via monkeypatching ``torch.cuda.is_available``.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from statspai.utils._torch_device import resolve_torch_device, torch_device_info


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("STATSPAI_TORCH_DEVICE", raising=False)
    yield


def test_default_is_cpu():
    assert resolve_torch_device().type == "cpu"


def test_empty_env_var_treated_as_unset(monkeypatch):
    monkeypatch.setenv("STATSPAI_TORCH_DEVICE", "")
    assert resolve_torch_device().type == "cpu"


def test_explicit_cpu_via_env(monkeypatch):
    monkeypatch.setenv("STATSPAI_TORCH_DEVICE", "cpu")
    assert resolve_torch_device().type == "cpu"


def test_per_call_override_beats_env(monkeypatch):
    monkeypatch.setenv("STATSPAI_TORCH_DEVICE", "cuda")
    # Per-call cpu wins even when env requests cuda.
    assert resolve_torch_device("cpu").type == "cpu"


def test_explicit_cuda_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("STATSPAI_TORCH_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="STATSPAI_TORCH_DEVICE"):
        resolve_torch_device()


def test_explicit_cuda_returns_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("STATSPAI_TORCH_DEVICE", "cuda")
    dev = resolve_torch_device()
    assert dev.type == "cuda"


def test_explicit_cuda_index_checks_device_count(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("STATSPAI_TORCH_DEVICE", "cuda:1")
    with pytest.raises(RuntimeError, match="only 1 device"):
        resolve_torch_device()


def test_auto_falls_back_to_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Force the MPS probe to return False even on Apple Silicon.
    import statspai.utils._torch_device as mod

    monkeypatch.setattr(mod, "_mps_available", lambda _t: False)
    monkeypatch.setenv("STATSPAI_TORCH_DEVICE", "auto")
    assert resolve_torch_device().type == "cpu"


def test_auto_prefers_cuda_over_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    import statspai.utils._torch_device as mod

    monkeypatch.setattr(mod, "_mps_available", lambda _t: True)
    monkeypatch.setenv("STATSPAI_TORCH_DEVICE", "auto")
    assert resolve_torch_device().type == "cuda"


def test_torch_device_info_is_string():
    info = torch_device_info()
    assert isinstance(info, str)
    assert "torch" in info


def test_fast_module_exposes_torch_device_info():
    from statspai.fast import torch_device_info as fast_info

    assert fast_info() == torch_device_info()
