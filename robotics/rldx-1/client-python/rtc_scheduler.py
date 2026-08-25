# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Client-owned action scheduling for RLDX-1 Real-Time Chunking (RTC)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class RTCProtocolError(ValueError):
    """The server response or announced RTC contract is inconsistent."""


class RTCLatePlan(RTCProtocolError):
    """A replacement plan missed the client's declared installation step."""


def _wire_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RTCProtocolError(f"{name} must be an integer, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise RTCProtocolError(f"{name} must be a finite integer, got {value!r}")
    return int(numeric)


@dataclass(frozen=True)
class InFlightRequest:
    request_id: int
    base_plan_id: int
    request_step: int
    install_step: int
    rtc_prefix_len: int


@dataclass(frozen=True)
class ActionPlan:
    plan_id: int
    base_plan_id: int
    origin_step: int
    actions: dict[str, np.ndarray]

    def action_at(self, control_step: int) -> dict[str, list[float]]:
        index = control_step - self.origin_step
        if index < 0 or index >= next(iter(self.actions.values())).shape[0]:
            raise RTCProtocolError(
                f"plan {self.plan_id} does not cover control_step={control_step}"
            )
        return {key: values[index].tolist() for key, values in self.actions.items()}

    def flat_prefix(
        self,
        *,
        start_step: int,
        length: int,
        action_order: tuple[str, ...],
    ) -> list[list[float]]:
        start = start_step - self.origin_step
        end = start + length
        horizon = next(iter(self.actions.values())).shape[0]
        if start < 0 or end > horizon:
            raise RTCProtocolError(
                f"plan {self.plan_id} cannot supply [{start_step}, {start_step + length})"
            )
        return np.concatenate(
            [self.actions[key][start:end] for key in action_order],
            axis=-1,
        ).tolist()


class RTCPlanScheduler:
    """Schedule RTC plans against the client's logical control-step cursor."""

    def __init__(
        self,
        *,
        action_horizon: int,
        exec_horizon: int,
        rtc_delay: int,
        action_order: list[str] | tuple[str, ...],
        action_dims: dict[str, int],
    ) -> None:
        self.action_horizon = int(action_horizon)
        self.exec_horizon = int(exec_horizon)
        self.rtc_delay = int(rtc_delay)
        self.action_order = tuple(str(key) for key in action_order)
        self.action_dims = {str(key): int(dim) for key, dim in action_dims.items()}

        if not (
            0 < self.rtc_delay
            <= self.exec_horizon
            <= self.action_horizon - self.rtc_delay
        ):
            raise RTCProtocolError(
                "RTC requires 0 < rtc_delay <= exec_horizon "
                "<= action_horizon - rtc_delay"
            )
        if not self.action_order or set(self.action_order) != set(self.action_dims):
            raise RTCProtocolError("action_order must contain every action_dims key once")
        if len(set(self.action_order)) != len(self.action_order):
            raise RTCProtocolError("action_order contains duplicate keys")
        if any(dim < 1 for dim in self.action_dims.values()):
            raise RTCProtocolError("every action dimension must be positive")

        self.next_request_id = 0
        self.reset()

    @classmethod
    def from_schema(cls, schema: dict[str, Any]) -> RTCPlanScheduler:
        raw_order = schema.get("action_order")
        raw_dims = schema.get("action_dims")
        if not isinstance(raw_order, (list, tuple)):
            raise RTCProtocolError("action_order must be a list")
        if not isinstance(raw_dims, dict):
            raise RTCProtocolError("action_dims must be an object")
        return cls(
            action_horizon=_wire_int(schema.get("action_horizon"), "action_horizon"),
            exec_horizon=_wire_int(schema.get("exec_horizon"), "exec_horizon"),
            rtc_delay=_wire_int(schema.get("rtc_delay"), "rtc_delay"),
            action_order=list(raw_order),
            action_dims={
                str(key): _wire_int(value, f"action_dims[{key!r}]")
                for key, value in raw_dims.items()
            },
        )

    def reset(self) -> None:
        """Return to safe hold; the next request will be a cold start."""

        self.current_step = 0
        self.active_plan: ActionPlan | None = None
        self.pending_plan: ActionPlan | None = None
        self.pending_install_step: int | None = None
        self.in_flight: InFlightRequest | None = None
        self.next_request_step: int | None = None

    @property
    def holding(self) -> bool:
        return self.active_plan is None

    def next_request(self) -> dict[str, Any] | None:
        """Return the request due at this control step, or ``None``."""

        if self.in_flight is not None or self.pending_plan is not None:
            return None

        if self.active_plan is None:
            request = InFlightRequest(
                request_id=self.next_request_id,
                base_plan_id=-1,
                request_step=0,
                install_step=0,
                rtc_prefix_len=0,
            )
            prefix: list[list[float]] = []
        else:
            if self.next_request_step is None or self.current_step < self.next_request_step:
                return None
            if self.current_step != self.next_request_step:
                raise RTCLatePlan(
                    f"missed RTC request step {self.next_request_step}; "
                    f"client is at {self.current_step}"
                )
            request = InFlightRequest(
                request_id=self.next_request_id,
                base_plan_id=self.active_plan.plan_id,
                request_step=self.current_step,
                install_step=self.current_step + self.rtc_delay,
                rtc_prefix_len=self.rtc_delay,
            )
            prefix = self.active_plan.flat_prefix(
                start_step=self.current_step,
                length=self.rtc_delay,
                action_order=self.action_order,
            )

        self.in_flight = request
        self.next_request_id += 1
        return {
            "request_id": request.request_id,
            "base_plan_id": request.base_plan_id,
            "install_step": request.install_step,
            "rtc_prefix_len": request.rtc_prefix_len,
            "action_prefix": prefix,
        }

    def accept(self, message: dict[str, Any]) -> ActionPlan:
        """Validate and stage one ``action_prediction`` response."""

        request = self.in_flight
        if request is None:
            raise RTCProtocolError("received an RTC action with no request in flight")

        echoed = {
            name: _wire_int(message.get(name), name)
            for name in (
                "request_id",
                "plan_id",
                "base_plan_id",
                "install_step",
                "rtc_prefix_len",
            )
        }
        if echoed["request_id"] != request.request_id:
            raise RTCProtocolError(
                f"response request_id={echoed['request_id']} does not match "
                f"in-flight request {request.request_id}"
            )
        if echoed["plan_id"] != request.request_id:
            raise RTCProtocolError("plan_id must equal request_id")
        for name in ("base_plan_id", "install_step", "rtc_prefix_len"):
            if echoed[name] != getattr(request, name):
                raise RTCProtocolError(
                    f"response {name}={echoed[name]} does not match request "
                    f"{getattr(request, name)}"
                )

        actions: dict[str, np.ndarray] = {}
        for key in self.action_order:
            try:
                values = np.asarray(message.get(key), dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise RTCProtocolError(f"action {key!r} is not a numeric array") from exc
            expected = (self.action_horizon, self.action_dims[key])
            if values.shape != expected:
                raise RTCProtocolError(
                    f"action {key!r} has shape {values.shape}; expected {expected}"
                )
            if not np.isfinite(values).all():
                raise RTCProtocolError(f"action {key!r} contains NaN or Inf")
            actions[key] = values

        cold_start = request.base_plan_id == -1
        if not cold_start and self.current_step > request.install_step:
            raise RTCLatePlan(
                f"plan {request.request_id} arrived at step {self.current_step}, "
                f"after install_step={request.install_step}"
            )

        plan = ActionPlan(
            plan_id=request.request_id,
            base_plan_id=request.base_plan_id,
            origin_step=request.request_step,
            actions=actions,
        )
        self.in_flight = None
        if cold_start:
            self.active_plan = plan
            self.next_request_step = self.exec_horizon
        else:
            self.pending_plan = plan
            self.pending_install_step = request.install_step
            self._activate_due_plan()
        return plan

    def action_for_current_step(self) -> dict[str, list[float]] | None:
        """Return the action to execute now, or ``None`` while safely holding."""

        self._activate_due_plan()
        if (
            self.in_flight is not None
            and self.in_flight.base_plan_id != -1
            and self.current_step >= self.in_flight.install_step
        ):
            raise RTCLatePlan(
                f"request {self.in_flight.request_id} missed "
                f"install_step={self.in_flight.install_step}"
            )
        if self.active_plan is None:
            return None
        return self.active_plan.action_at(self.current_step)

    def advance(self) -> None:
        """Advance one control step after the returned action was executed."""

        if self.active_plan is None:
            raise RTCProtocolError("cannot advance while holding without an active plan")
        self.current_step += 1

    def _activate_due_plan(self) -> None:
        if self.pending_plan is None or self.pending_install_step is None:
            return
        if self.current_step < self.pending_install_step:
            return
        if self.current_step > self.pending_install_step:
            raise RTCLatePlan(
                f"missed plan installation at step {self.pending_install_step}"
            )
        self.active_plan = self.pending_plan
        self.pending_plan = None
        self.pending_install_step = None
        self.next_request_step = self.active_plan.origin_step + self.exec_horizon
