# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest

from srtctl.core.job_ports import SLOT_COUNT, JobPortPlan
from srtctl.ports import HTTP_PORTS


def test_all_slots_and_listener_families_have_disjoint_reservations():
    reserved = set()
    for slot in range(1, SLOT_COUNT):
        plan = JobPortPlan(slot)
        listeners = set(plan.to_dict()["fixed"].values())
        for base, size in plan.worker_ranges().values():
            block = set(range(base, base + size))
            assert not listeners.intersection(block)
            listeners.update(block)
        assert not reserved.intersection(listeners)
        assert max(listeners) < 49151
        reserved.update(listeners)


def test_neighbor_jobs_separate_ports_and_never_select_legacy_slot():
    with patch.dict("os.environ", {}, clear=True):
        for job in range(100, 160):
            left = JobPortPlan.from_job_id(str(job))
            right = JobPortPlan.from_job_id(str(job + 1))
            assert left.slot != right.slot
            assert left.fixed("frontend") != right.fixed("frontend")


def test_range_exhaustion_fails_instead_of_entering_the_next_jobs_slot():
    allocator = JobPortPlan(1).allocator()
    for _ in range(16):
        allocator.next(HTTP_PORTS, "node0")
    with pytest.raises(ValueError, match="range exhausted"):
        allocator.next(HTTP_PORTS, "node0")


def test_slot_override_and_invalid_ids_are_explicit():
    with patch.dict("os.environ", {"SRTCTL_PORT_SLOT": "3"}):
        assert JobPortPlan.from_job_id("123").slot == 3
    with patch.dict("os.environ", {"SRTCTL_PORT_SLOT": "0"}), pytest.raises(ValueError):
        JobPortPlan.from_job_id("123")
    with patch.dict("os.environ", {}, clear=True), pytest.raises(ValueError):
        JobPortPlan.from_job_id("unknown")
