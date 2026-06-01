# Changelog

## 0.5.1

### Fixed
- Wrap `aiohttp.ClientError` and `asyncio.TimeoutError` from both
  `send_command`'s REST POST and the upstream `_auth.get_client()` refresh in
  `ConnectionError` (an `AquariteError` subclass), so transport failures
  surface as the library's documented domain error rather than leaking aiohttp
  internals to callers.

## 0.5.0

### Added
- `AquariteClient.get_value()` now normalises known fields to native Python
  types regardless of how the Hayward cloud encodes them. The Aquarite
  firmware returns some numeric scalars as strings (`"747"`, `"600"`) and
  uses `0`/`1` ints for fields that are semantically booleans (`hasPH`,
  `cover_enabled`, …); the exact encoding can also vary between firmware
  revisions. Consumers no longer need defensive `int(str(...))` or
  `_coerce_to_bool` helpers.
- Per-path coercion is driven by a typed map in
  `aioaquarite._coercion._TYPE_MAP`, with wildcard support for module
  sub-keys. Adding a new path is a one-line change.

### Behaviour
- Missing keys, `None` values, and unparseable data all return the
  caller-supplied `default` (preserving the existing `default=None`
  contract). Unparseable data additionally logs a `WARNING` on the
  `aioaquarite._coercion` logger so firmware drift is visible without
  crashing the consumer.
- Unmapped paths are returned unchanged — no behavioural change for
  fields not in the map.
- The raw pool data cache used by `set_value()` is untouched; commands
  still round-trip the original on-wire encoding.

### Fields newly typed
- **Coerced from string → int**: `modules.*.current`,
  `modules.*.status.value`, `modules.*.status.low_value`,
  `modules.*.status.high_value`, `modules.io.status`,
  `filtration.intel.time`, `relays.*.gpio`, `relays.*.*.gpio`.
- **Coerced from 0/1 int (or string) → bool**: `main.has*`,
  `main.hide*`, `main.networkPresent`, `main.LEDPulse`,
  `main.FWU_enabled`, `hidro.hasHidroControl`, `hidro.cover`,
  `hidro.cover_enabled`, `hidro.cloration_enabled`,
  `hidro.temperature_enabled`, `hidro.fl1`, `hidro.fl2`,
  `hidro.is_electrolysis`, `hidro.reduction`, `hidro.low`,
  `filtration.hasSmart`, `filtration.smart.freeze`,
  `filtration.timerVel1/2/3`, `relays.*.info.onoff`,
  `relays.*.info.polarity`, `relays.*.info.manAutoTemp`,
  `relays.*.info.signal`, `modules.*.pump_status`,
  `modules.*.pump_high_on`, `modules.*.pump_low_on`,
  `form.active`, `present`, `isAWS`.
- **Float**: `main.temperature`, `form.lat`, `form.lng`.

`filtration.hasHeat` is multi-state (`0`/`1`/`2`) and stays `int`, not
`bool`.

## 0.4.0

- Initial public release.
