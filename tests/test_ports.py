# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for job-scoped runtime ports."""

from unittest.mock import patch

import pytest

from srtctl.ports import PORT_SLOT_COUNT, PORT_SLOT_STRIDE, PortPlan


def test_default_plan_preserves_legacy_ports():
    plan = PortPlan.default()

    assert plan.slot == 0
    assert plan.etcd_client_port == 2379
    assert plan.nats_port == 4222
    assert plan.frontend_public_port == 8000
    assert plan.dyn_system_port_base == 8081
    assert plan.node_port_allocator().base_fpm_port == 20380


def test_job_id_selects_a_shared_nonzero_offset_for_every_port_family():
    plan = PortPlan.from_job_id("14084685")
    expected_slot = 14084685 % PORT_SLOT_COUNT

    assert plan.slot == expected_slot
    assert plan.offset == expected_slot * PORT_SLOT_STRIDE
    assert plan.frontend_public_port == 8000 + plan.offset
    assert plan.dyn_system_port_base == 8081 + plan.offset
    assert plan.nats_port == 4222 + plan.offset

    allocator = plan.node_port_allocator()
    assert allocator.base_http_port == 30000 + plan.offset
    assert allocator.base_kv_events_port == 5550 + plan.offset
    assert allocator.base_nixl_port == 6550 + plan.offset
    assert allocator.base_fpm_port == 20380 + (expected_slot * 1024)


def test_neighboring_jobs_select_different_plans():
    first = PortPlan.from_job_id("14084684")
    second = PortPlan.from_job_id("14084685")

    assert first.slot != second.slot
    assert first.frontend_public_port != second.frontend_public_port
    assert first.dyn_system_port_base != second.dyn_system_port_base
    assert first.node_port_allocator().base_fpm_port != second.node_port_allocator().base_fpm_port


def test_explicit_slot_override_is_honored():
    with patch.dict("os.environ", {"SRTCTL_PORT_SLOT": "7"}):
        plan = PortPlan.from_job_id("14084685")

    assert plan.slot == 7


def test_invalid_slot_override_fails_early():
    with (
        patch.dict("os.environ", {"SRTCTL_PORT_SLOT": str(PORT_SLOT_COUNT)}),
        pytest.raises(ValueError, match="Port slot"),
    ):
        PortPlan.from_job_id("14084685")
