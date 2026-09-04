# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Regression test for lossless client-prefix normalization in vendored RLDX."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is installed in the model image")

from rldx.policy.policy_runtime import PolicyRuntime  # noqa: E402


class FakeStateActionProcessor:
    def __init__(self) -> None:
        self.modality_configs = {
            "test": {
                "action": SimpleNamespace(modality_keys=["joint"]),
            }
        }
        self.norm_params = {
            "test": {
                "action": {
                    "joint": {"min": np.zeros(3, dtype=np.float32)},
                }
            }
        }
        self.clip_override = None

    def apply_action(
        self,
        action,
        embodiment_tag,
        state=None,
        *,
        clip_outliers=None,
    ):
        self.clip_override = clip_outliers
        return action


class FakeRegistry:
    def resolve_sids(self, _sids, _batch_size):
        return ["default"]

    def invalidate_rtc(self, _sids, _reset_memory):
        return None


def test_client_prefix_disables_training_time_outlier_clipping():
    sap = FakeStateActionProcessor()
    runtime = PolicyRuntime.__new__(PolicyRuntime)
    runtime._rtc_enabled = True
    runtime.registry = FakeRegistry()
    runtime.rtc_inference_delay = 1
    runtime.rtc_exec_horizon = 8
    runtime.rtc_inference_mode = "trained"
    runtime.processor = SimpleNamespace(state_action_processor=sap)
    runtime.embodiment_tag = SimpleNamespace(value="test")
    runtime.model = SimpleNamespace(
        action_model=SimpleNamespace(action_dim=3),
        device=torch.device("cpu"),
    )
    runtime.verbose = False

    request = SimpleNamespace(
        sids=None,
        action_prefix=np.asarray([[1.5, 0.0, -1.5]], dtype=np.float32),
        rtc_prefix_len=1,
    )
    collated = {}
    runtime._inject_rtc_prefix(request, collated, 1, None)

    assert sap.clip_override is False
    assert torch.equal(
        collated["action_prefix"].float(),
        torch.tensor([[[1.5, 0.0, -1.5]]]),
    )
