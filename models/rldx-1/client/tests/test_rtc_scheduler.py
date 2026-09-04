# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Tests for the cookbook RLDX-1 RTC plan scheduler."""

from __future__ import annotations

import unittest

from rtc_scheduler import RTCLatePlan, RTCPlanScheduler, RTCProtocolError

ACTION_ORDER = ["arm", "gripper"]
ACTION_DIMS = {"arm": 2, "gripper": 1}


def scheduler() -> RTCPlanScheduler:
    return RTCPlanScheduler(
        action_horizon=8,
        exec_horizon=3,
        rtc_delay=2,
        action_order=ACTION_ORDER,
        action_dims=ACTION_DIMS,
    )


def response(request: dict, plan_id: int, offset: float = 0.0) -> dict:
    return {
        "request_id": request["request_id"],
        "plan_id": plan_id,
        "base_plan_id": request["base_plan_id"],
        "install_step": request["install_step"],
        "rtc_prefix_len": request["rtc_prefix_len"],
        "arm": [[offset + step, offset + step + 0.5] for step in range(8)],
        "gripper": [[offset + 100 + step] for step in range(8)],
    }


class RTCPlanSchedulerTest(unittest.TestCase):
    def test_cold_start_holds_until_the_first_plan(self) -> None:
        rtc = scheduler()
        first = rtc.next_request()
        self.assertEqual(
            first,
            {
                "request_id": 0,
                "base_plan_id": -1,
                "install_step": 0,
                "rtc_prefix_len": 0,
                "action_prefix": [],
            },
        )
        self.assertIsNone(rtc.action_for_current_step())

        rtc.accept(response(first, plan_id=0))
        self.assertEqual(rtc.action_for_current_step()["arm"], [0.0, 0.5])

    def test_replacement_request_prefix_and_installation(self) -> None:
        rtc = scheduler()
        first = rtc.next_request()
        rtc.accept(response(first, plan_id=0))

        for _ in range(3):
            rtc.action_for_current_step()
            rtc.advance()

        second = rtc.next_request()
        self.assertEqual(second["base_plan_id"], 0)
        self.assertEqual(second["install_step"], 5)
        self.assertEqual(second["rtc_prefix_len"], 2)
        self.assertEqual(
            second["action_prefix"],
            [[3.0, 3.5, 103.0], [4.0, 4.5, 104.0]],
        )

        rtc.accept(response(second, plan_id=1, offset=1000.0))
        self.assertEqual(rtc.action_for_current_step()["arm"], [3.0, 3.5])
        rtc.advance()
        self.assertEqual(rtc.action_for_current_step()["arm"], [4.0, 4.5])
        rtc.advance()
        self.assertEqual(rtc.current_step, 5)
        self.assertEqual(rtc.action_for_current_step()["arm"], [1002.0, 1002.5])
        rtc.advance()

        third = rtc.next_request()
        self.assertEqual(third["base_plan_id"], 1)
        self.assertEqual(third["install_step"], 8)
        self.assertEqual(
            third["action_prefix"],
            [[1003.0, 1003.5, 1103.0], [1004.0, 1004.5, 1104.0]],
        )

    def test_missing_deadline_resets_instead_of_splicing_late(self) -> None:
        rtc = scheduler()
        first = rtc.next_request()
        rtc.accept(response(first, plan_id=0))
        for _ in range(3):
            rtc.action_for_current_step()
            rtc.advance()
        second = rtc.next_request()
        for _ in range(2):
            rtc.action_for_current_step()
            rtc.advance()

        with self.assertRaises(RTCLatePlan):
            rtc.action_for_current_step()
        rtc.reset()
        self.assertTrue(rtc.holding)
        cold_restart = rtc.next_request()
        self.assertEqual(cold_restart["base_plan_id"], -1)
        self.assertEqual(cold_restart["request_id"], 2)

        # A response from the abandoned chain cannot be accepted after reset.
        with self.assertRaisesRegex(RTCProtocolError, "request_id"):
            rtc.accept(response(second, plan_id=1))

    def test_rejects_mismatched_response_metadata(self) -> None:
        rtc = scheduler()
        first = rtc.next_request()
        bad = response(first, plan_id=9)
        with self.assertRaisesRegex(RTCProtocolError, "plan_id"):
            rtc.accept(bad)

    def test_rejects_bad_action_shape_and_nonfinite_values(self) -> None:
        rtc = scheduler()
        first = rtc.next_request()
        bad_shape = response(first, plan_id=0)
        bad_shape["arm"] = [[0.0, 0.0]]
        with self.assertRaisesRegex(RTCProtocolError, "shape"):
            rtc.accept(bad_shape)

        rtc.reset()
        first = rtc.next_request()
        bad_value = response(first, plan_id=first["request_id"])
        bad_value["gripper"][0][0] = float("nan")
        with self.assertRaisesRegex(RTCProtocolError, "NaN or Inf"):
            rtc.accept(bad_value)

        rtc.reset()
        first = rtc.next_request()
        bad_array = response(first, plan_id=first["request_id"])
        bad_array["arm"] = [[0.0], [0.0, 1.0]]
        with self.assertRaisesRegex(RTCProtocolError, "numeric array"):
            rtc.accept(bad_array)

    def test_schema_validation(self) -> None:
        with self.assertRaises(RTCProtocolError):
            RTCPlanScheduler(
                action_horizon=8,
                exec_horizon=7,
                rtc_delay=2,
                action_order=ACTION_ORDER,
                action_dims=ACTION_DIMS,
            )
        with self.assertRaisesRegex(RTCProtocolError, "action_order"):
            RTCPlanScheduler.from_schema(
                {
                    "action_horizon": 8,
                    "exec_horizon": 3,
                    "rtc_delay": 2,
                    "action_order": "arm",
                    "action_dims": ACTION_DIMS,
                }
            )
        with self.assertRaisesRegex(RTCProtocolError, "action_dims"):
            RTCPlanScheduler.from_schema(
                {
                    "action_horizon": 8,
                    "exec_horizon": 3,
                    "rtc_delay": 2,
                    "action_order": ACTION_ORDER,
                    "action_dims": {"arm": 2, "gripper": float("nan")},
                }
            )


if __name__ == "__main__":
    unittest.main()
