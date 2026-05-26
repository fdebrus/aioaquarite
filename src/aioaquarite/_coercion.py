"""Per-path type coercion for raw Firestore pool payloads.

The Hayward Aquarite cloud encodes some scalar fields as strings
(e.g. ``modules.ph.current = "747"``) and uses 0/1 ints for fields that
are semantically booleans (e.g. ``main.hasPH``). On top of that, the
exact encoding varies between firmware revisions: the same field may
arrive as ``"0"`` on one pool and ``0`` on another. This module
centralises normalisation so library consumers see Pythonic types via
:meth:`AquariteClient.get_value`.

Path patterns may contain ``*`` to match a single dot-separated segment.
Literal paths take precedence over wildcard patterns when both match.
"""

from __future__ import annotations

import logging
from typing import Any, Final, Literal

_LOGGER = logging.getLogger(__name__)

ExpectedType = Literal["int", "float", "bool"]

_TYPE_MAP: Final[dict[str, ExpectedType]] = {
    # ── top-level ──
    "present": "bool",
    "isAWS": "bool",
    "createdAt": "int",
    "updatedAt": "int",
    "company": "int",
    # ── form ──
    "form.lat": "float",
    "form.lng": "float",
    "form.active": "bool",
    # ── main ──
    "main.hasPH": "bool",
    "main.hasRX": "bool",
    "main.hasUV": "bool",
    "main.hasIO": "bool",
    "main.hasCL": "bool",
    "main.hasCD": "bool",
    "main.hasHidro": "bool",
    "main.hasWifi": "bool",
    "main.hasLCD": "bool",
    "main.hasLED": "bool",
    "main.hasLinked": "bool",
    "main.hasLinkedAuto": "bool",
    "main.hasBackwash": "bool",
    "main.hideFiltration": "bool",
    "main.hideLighting": "bool",
    "main.hideTemperature": "bool",
    "main.hideRelays": "bool",
    "main.networkPresent": "bool",
    "main.LEDPulse": "bool",
    "main.FWU_enabled": "bool",
    "main.RSSI": "int",
    "main.version": "int",
    "main.wifiVersion": "int",
    "main.localTime": "int",
    "main.temperature": "float",
    # ── modules ──
    "modules.*.current": "int",
    "modules.*.status.value": "int",
    "modules.*.status.low_value": "int",
    "modules.*.status.high_value": "int",
    "modules.io.status": "int",
    "modules.*.tank": "int",
    "modules.*.pump_status": "bool",
    "modules.*.pump_high_on": "bool",
    "modules.*.pump_low_on": "bool",
    "modules.*.al3": "int",
    "modules.*.al4": "int",
    "modules.*.activation": "int",
    "modules.*.level": "int",
    "modules.*.partial": "int",
    "modules.*.total": "int",
    # ── hidro ──
    "hidro.hasHidroControl": "bool",
    "hidro.cover": "bool",
    "hidro.cover_enabled": "bool",
    "hidro.cloration_enabled": "bool",
    "hidro.temperature_enabled": "bool",
    "hidro.fl1": "bool",
    "hidro.fl2": "bool",
    "hidro.is_electrolysis": "bool",
    "hidro.reduction": "bool",
    "hidro.low": "bool",
    "hidro.al4": "int",
    "hidro.level": "int",
    "hidro.maxAllowedValue": "int",
    "hidro.temperature_value": "int",
    "hidro.cellPartialTime": "int",
    "hidro.cellTotalTime": "int",
    "hidro.control": "int",
    "hidro.current": "int",
    # ── filtration ──
    "filtration.intel.time": "int",
    "filtration.intel.temp": "int",
    "filtration.hasSmart": "bool",
    "filtration.hasHeat": "int",
    "filtration.smart.freeze": "bool",
    "filtration.smart.tempHigh": "int",
    "filtration.smart.tempMin": "int",
    "filtration.timerVel1": "bool",
    "filtration.timerVel2": "bool",
    "filtration.timerVel3": "bool",
    "filtration.status": "int",
    "filtration.pumpType": "int",
    "filtration.manVel": "int",
    "filtration.mode": "int",
    "filtration.heating.tempHi": "int",
    "filtration.heating.temp": "int",
    "filtration.heating.clima": "int",
    "filtration.interval1.from": "int",
    "filtration.interval1.to": "int",
    "filtration.interval2.from": "int",
    "filtration.interval2.to": "int",
    "filtration.interval3.from": "int",
    "filtration.interval3.to": "int",
    # ── relays ──
    "relays.*.gpio": "int",
    "relays.*.*.gpio": "int",
    "relays.*.info.key": "int",
    "relays.*.info.onoff": "bool",
    "relays.*.info.polarity": "bool",
    "relays.*.info.manAutoTemp": "bool",
    "relays.*.info.signal": "bool",
    "relays.*.info.status": "int",
    "relays.*.info.from": "int",
    "relays.*.info.from2": "int",
    "relays.*.info.to": "int",
    "relays.*.info.to2": "int",
    "relays.*.info.freq": "int",
    "relays.*.info.freq2": "int",
    "relays.*.info.delay": "int",
    "relays.*.info.tiempoOn": "int",
    "relays.filtration.heating.status": "int",
    # ── backwash ──
    "backwash.status": "int",
    "backwash.interval": "int",
    "backwash.frequency": "int",
    "backwash.remainingTime": "int",
    "backwash.startAt": "int",
    "backwash.mode": "int",
    # ── light ──
    "light.status": "int",
    "light.from": "int",
    "light.to": "int",
    "light.freq": "int",
    "light.mode": "int",
}

_SENTINEL: Final = object()


def expected_type(path: str) -> ExpectedType | None:
    """Return the expected type for ``path``, or ``None`` if no rule applies.

    Literal paths take precedence over wildcard patterns.
    """
    literal = _TYPE_MAP.get(path)
    if literal is not None:
        return literal
    segs = path.split(".")
    for pattern, t in _TYPE_MAP.items():
        if "*" not in pattern:
            continue
        psegs = pattern.split(".")
        if len(psegs) != len(segs):
            continue
        if all(p == "*" or p == s for p, s in zip(psegs, segs)):
            return t
    return None


def _coerce(value: Any, expected: ExpectedType) -> Any:
    """Coerce ``value`` to ``expected``. Returns ``_SENTINEL`` on failure."""
    if expected == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("0", "false"):
                return False
            if s in ("1", "true"):
                return True
        return _SENTINEL

    if expected == "int":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return _SENTINEL
        if isinstance(value, str):
            s = value.strip()
            try:
                return int(s)
            except ValueError:
                try:
                    f = float(s)
                except ValueError:
                    return _SENTINEL
                if f.is_integer():
                    return int(f)
                return _SENTINEL
        return _SENTINEL

    if expected == "float":
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return _SENTINEL
        return _SENTINEL

    return _SENTINEL


def normalise(path: str, value: Any, default: Any) -> Any:
    """Apply per-path coercion.

    - ``value is None`` → return ``default``.
    - No rule for ``path`` → return ``value`` unchanged.
    - Unparseable for the expected type → log WARNING and return ``default``.
    """
    if value is None:
        return default
    et = expected_type(path)
    if et is None:
        return value
    result = _coerce(value, et)
    if result is _SENTINEL:
        _LOGGER.warning(
            "aioaquarite: cannot coerce %r at path %s to %s; returning default",
            value,
            path,
            et,
        )
        return default
    return result
