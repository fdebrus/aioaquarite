"""Tests for AquariteClient.get_value type coercion."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from aioaquarite import AquariteClient
from aioaquarite._coercion import _TYPE_MAP, expected_type

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pool_data.json"


@pytest.fixture(scope="module")
def pool_data() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text())


# ── string-encoded numerics get coerced ────────────────────────────────────

def test_ph_current_string_is_coerced_to_int(pool_data: dict[str, Any]) -> None:
    val = AquariteClient.get_value(pool_data, "modules.ph.current")
    assert val == 747
    assert type(val) is int


def test_ph_status_thresholds_string_are_coerced(pool_data: dict[str, Any]) -> None:
    assert AquariteClient.get_value(pool_data, "modules.ph.status.low_value") == 650
    assert AquariteClient.get_value(pool_data, "modules.ph.status.high_value") == 751


def test_filtration_intel_time_string_is_coerced(pool_data: dict[str, Any]) -> None:
    val = AquariteClient.get_value(pool_data, "filtration.intel.time")
    assert val == 600
    assert type(val) is int


def test_modules_status_value_string_is_coerced(pool_data: dict[str, Any]) -> None:
    assert AquariteClient.get_value(pool_data, "modules.cd.status.value") == 5000
    assert AquariteClient.get_value(pool_data, "modules.cl.status.value") == 100
    # firmware-variant: rx.status.value already int — must round-trip
    assert AquariteClient.get_value(pool_data, "modules.rx.status.value") == 700


def test_modules_io_status_string_is_coerced(pool_data: dict[str, Any]) -> None:
    val = AquariteClient.get_value(pool_data, "modules.io.status")
    assert val == 8192
    assert type(val) is int


def test_relay_gpio_strings_coerced_to_int(pool_data: dict[str, Any]) -> None:
    assert AquariteClient.get_value(pool_data, "relays.uv.gpio") == 0
    assert AquariteClient.get_value(pool_data, "relays.io.gpio") == 182
    assert AquariteClient.get_value(pool_data, "relays.ph.acid.gpio") == 1
    assert AquariteClient.get_value(pool_data, "relays.ph.base.gpio") == 0
    assert AquariteClient.get_value(pool_data, "relays.filtration.heating.gpio") == 7
    # already int — must round-trip
    assert AquariteClient.get_value(pool_data, "relays.filtration.gpio") == 2
    assert AquariteClient.get_value(pool_data, "relays.light.gpio") == 3


# ── 0/1 ints get coerced to bool ──────────────────────────────────────────

def test_main_has_flags_are_bool(pool_data: dict[str, Any]) -> None:
    val = AquariteClient.get_value(pool_data, "main.hasPH")
    assert val is True
    assert type(val) is bool
    assert AquariteClient.get_value(pool_data, "main.hasUV") is False
    assert AquariteClient.get_value(pool_data, "main.hasHidro") is True


def test_main_hide_flags_are_bool(pool_data: dict[str, Any]) -> None:
    assert AquariteClient.get_value(pool_data, "main.hideFiltration") is False
    assert AquariteClient.get_value(pool_data, "main.hideRelays") is False


def test_already_bool_passes_through(pool_data: dict[str, Any]) -> None:
    # FWU_enabled and is_electrolysis come back as real bools in this firmware
    assert AquariteClient.get_value(pool_data, "main.FWU_enabled") is False
    assert AquariteClient.get_value(pool_data, "hidro.is_electrolysis") is True
    assert AquariteClient.get_value(pool_data, "present") is True
    assert AquariteClient.get_value(pool_data, "isAWS") is True


def test_hidro_flags_are_bool(pool_data: dict[str, Any]) -> None:
    assert AquariteClient.get_value(pool_data, "hidro.cover_enabled") is False
    assert AquariteClient.get_value(pool_data, "hidro.cloration_enabled") is False
    assert AquariteClient.get_value(pool_data, "hidro.fl1") is False
    assert AquariteClient.get_value(pool_data, "hidro.fl2") is True


def test_relay_info_onoff_polarity_are_bool(pool_data: dict[str, Any]) -> None:
    val = AquariteClient.get_value(pool_data, "relays.relay1.info.onoff")
    assert val is False
    assert type(val) is bool
    assert AquariteClient.get_value(pool_data, "relays.relay2.info.polarity") is False


def test_pump_flags_are_bool(pool_data: dict[str, Any]) -> None:
    assert AquariteClient.get_value(pool_data, "modules.ph.pump_high_on") is False
    assert AquariteClient.get_value(pool_data, "modules.ph.pump_low_on") is False
    assert AquariteClient.get_value(pool_data, "modules.rx.pump_status") is False


def test_filtration_multistate_int_not_bool(pool_data: dict[str, Any]) -> None:
    # hasHeat is a multi-state int (0/1/2), not a boolean.
    val = AquariteClient.get_value(pool_data, "filtration.hasHeat")
    assert val == 2
    assert type(val) is int


# ── floats and passthrough ────────────────────────────────────────────────

def test_temperature_is_float(pool_data: dict[str, Any]) -> None:
    val = AquariteClient.get_value(pool_data, "main.temperature")
    assert val == 30.3
    assert type(val) is float


def test_unmapped_string_passes_through(pool_data: dict[str, Any]) -> None:
    # modules.ph.type → "ACID" — no rule, raw passthrough.
    assert AquariteClient.get_value(pool_data, "modules.ph.type") == "ACID"
    assert AquariteClient.get_value(pool_data, "hidro.measure") == "gr/h"
    assert AquariteClient.get_value(pool_data, "form.country") == "BE"


def test_unmapped_int_passes_through(pool_data: dict[str, Any]) -> None:
    # form.name has no rule and is a string
    assert AquariteClient.get_value(pool_data, "form.name") == "Home"


# ── default semantics preserved ────────────────────────────────────────────

def test_missing_path_returns_default(pool_data: dict[str, Any]) -> None:
    assert AquariteClient.get_value(pool_data, "does.not.exist") is None
    assert AquariteClient.get_value(pool_data, "does.not.exist", default=42) == 42


def test_empty_data_returns_default() -> None:
    assert AquariteClient.get_value({}, "modules.ph.current") is None
    assert AquariteClient.get_value({}, "modules.ph.current", default=0) == 0


def test_traverse_through_non_dict_returns_default(pool_data: dict[str, Any]) -> None:
    # modules.ph.current is a leaf — walking past it returns default.
    assert (
        AquariteClient.get_value(pool_data, "modules.ph.current.something")
        is None
    )


def test_explicit_none_in_data_returns_default() -> None:
    data = {"modules": {"ph": {"current": None}}}
    assert AquariteClient.get_value(data, "modules.ph.current") is None
    assert AquariteClient.get_value(data, "modules.ph.current", default=0) == 0


# ── unparseable values: default + WARNING ──────────────────────────────────

def test_unparseable_string_for_int_returns_default_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = {"modules": {"ph": {"current": "not-a-number"}}}
    with caplog.at_level(logging.WARNING, logger="aioaquarite._coercion"):
        val = AquariteClient.get_value(data, "modules.ph.current", default=-1)
    assert val == -1
    assert any(
        "cannot coerce" in r.message and "modules.ph.current" in r.message
        for r in caplog.records
    )


def test_unparseable_string_for_bool_returns_default_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = {"main": {"hasPH": "maybe"}}
    with caplog.at_level(logging.WARNING, logger="aioaquarite._coercion"):
        val = AquariteClient.get_value(data, "main.hasPH")
    assert val is None
    assert any("cannot coerce" in r.message for r in caplog.records)


def test_unexpected_dict_for_scalar_returns_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A firmware that wrapped a scalar in a dict — must not crash.
    data = {"modules": {"io": {"status": {"unexpected": 1}}}}
    with caplog.at_level(logging.WARNING, logger="aioaquarite._coercion"):
        val = AquariteClient.get_value(data, "modules.io.status")
    assert val is None


def test_int_for_bool_outside_01_returns_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Conservative: only 0/1 map to bool. A 2 at a bool path is suspicious.
    data = {"main": {"hasPH": 2}}
    with caplog.at_level(logging.WARNING, logger="aioaquarite._coercion"):
        val = AquariteClient.get_value(data, "main.hasPH")
    assert val is None


# ── wildcard path matching ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "path, expected",
    [
        ("modules.ph.current", "int"),
        ("modules.cd.current", "int"),
        ("modules.cl.current", "int"),
        ("modules.rx.current", "int"),
        ("modules.cd.status.value", "int"),
        ("modules.ph.status.low_value", "int"),
        ("relays.relay1.info.onoff", "bool"),
        ("relays.relay4.info.signal", "bool"),
        ("relays.ph.base.gpio", "int"),
        ("relays.ph.acid.gpio", "int"),
        ("relays.uv.gpio", "int"),
    ],
)
def test_wildcard_paths_resolve(path: str, expected: str) -> None:
    assert expected_type(path) == expected


def test_literal_precedence_over_wildcard() -> None:
    # modules.io.status is a literal (scalar bitmap); the wildcards in the
    # map only target modules.*.status.{value,low_value,high_value}, so no
    # ambiguity in this map — but verify the lookup order anyway.
    assert expected_type("modules.io.status") == "int"


def test_unknown_path_has_no_expected_type() -> None:
    assert expected_type("does.not.exist") is None
    assert expected_type("modules.ph.type") is None  # string field, not mapped


# ── sanity: every mapped path in the fixture coerces cleanly ──────────────

def test_full_fixture_no_coercion_warnings(
    pool_data: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Walking every mapped path against the real-shape fixture must not warn."""
    # Map of pattern → resolved paths present in the fixture.
    resolved: list[str] = []
    for pattern in _TYPE_MAP:
        if "*" not in pattern:
            resolved.append(pattern)
            continue
        for path in _enumerate_matching_paths(pool_data, pattern.split(".")):
            resolved.append(path)

    with caplog.at_level(logging.WARNING, logger="aioaquarite._coercion"):
        for p in resolved:
            AquariteClient.get_value(pool_data, p)
    assert not caplog.records, (
        "unexpected coercion warnings:\n"
        + "\n".join(r.message for r in caplog.records)
    )


def _enumerate_matching_paths(
    data: Any, pattern_segs: list[str], prefix: list[str] | None = None
) -> list[str]:
    """Yield concrete dot-paths in ``data`` matching a wildcard pattern."""
    prefix = prefix or []
    if not pattern_segs:
        return [".".join(prefix)] if not isinstance(data, dict) else []
    head, *rest = pattern_segs
    if not isinstance(data, dict):
        return []
    if head == "*":
        out: list[str] = []
        for k, v in data.items():
            out.extend(_enumerate_matching_paths(v, rest, prefix + [k]))
        return out
    if head in data:
        return _enumerate_matching_paths(data[head], rest, prefix + [head])
    return []
