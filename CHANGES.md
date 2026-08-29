# Changelog

## 0.9.1

### Fixed
- Serialise concurrent commands targeting the same branch of a pool.
  The payload is rebuilt from the stored document, and the document was
  only updated after the send completed, so two callers writing
  different fields of one branch at the same time each sent a branch
  that predated the other and silently reverted it. Commands for the
  same pool and branch now run under a lock covering payload
  construction, the send, and the cache update.

## 0.9.0

### Added
- Optional ``on_health`` callback on the resilient subscriptions (and the
  ``subscribe_*_resilient`` factories): reports connection-state
  transitions — ``on_health(False)`` when the connection is lost,
  ``on_health(True)`` once re-established. Fires on transitions only,
  never for ``aclose()``, and is invoked from the supervisor's event
  loop (unlike the data callback, which runs on the Firestore thread).
  A raising callback is logged and never kills the supervisor.
- ``healthy`` property on the resilient subscriptions.

### Fixed
- After a failed reconnect, the supervisor now re-establishes the watch
  on the next healthy tick. Previously the watch stayed dead if the
  network recovered between ticks and no token refresh occurred, leaving
  the subscription silently disconnected forever.

## 0.8.0

### Added
- `AquariteClient.get_pool_stats(pool_id, type_, period)` calls the
  Hayward `getStats` cloud function and returns the stored sample series
  for a metric (`ph`, `rx`, `temp`, `cl`, `cd`, `filtration`, `aux1`-`aux4`,
  and the hardware-conditional `light` / `production` / `salt`).
  Returns the raw decoded payload — a list of series, each series a
  list of `{"field": <value>, "seconds": <utc_unix>}` dicts. `period`
  is required by the cloud function (requests without it are rejected
  with HTTP 405) but appears to be ignored semantically — the response
  covers ~30 days regardless of the value passed. Field encodings are
  documented on the method docstring.
- `AquariteClient.get_server_date()` calls the unauthenticated
  `getServerDate` cloud function and returns `{"date": "YYMMDD"}` — handy
  for clock-drift checks against the Hayward backend.
- `aioaquarite.const.DEFAULT_HTTP_TIMEOUT` (`20` seconds) so the new
  helpers and any future REST calls share one knob.

### Changed
- `get_pool_stats` now refreshes the auth token via
  `AquariteAuth.get_client()` before building its request headers,
  matching every other authenticated method (`send_command`,
  `get_pools`, `fetch_pool_data`, `subscribe_pool`). It previously read
  `self._auth.tokens['idToken']` directly, so a token that aged past
  expiry mid-session would send a stale token and surface as an
  intermittent `CommandError` instead of transparently refreshing.
- `get_pool_stats` and `get_server_date` now wrap `aiohttp.ClientError`
  and `asyncio.TimeoutError` into `ConnectionError`, matching
  `send_command`'s error contract, so callers only ever need to catch
  `AquariteError` (and subclasses) for transport failures.

Original endpoint reverse-engineering and tests by @aeddi (#6).

## 0.7.0

### Added
- `AquariteClient.set_values(pool_id, updates)` — write several values
  of one command branch as a single cloud command. All paths must share
  the same command branch (same top-level key, and same second-level
  key for deep 4+ segment paths); mixing branches raises `ValueError`.
  This is the primitive needed for writes that must land atomically,
  such as a light mode + status pair.

### Changed
- `set_value` is now a thin wrapper over `set_values`.
- On a successful send, `set_value`/`set_values` now mirror the written
  values into the stored pool data. Previously the stored document only
  refreshed on the next Firestore snapshot, so two quick writes to the
  same branch built the second payload from a stale document and
  silently reverted the first write.

## 0.6.1

### Added
- `AquariteClient.subscribe_user_pools(callback)` — push-based
  subscription on the `users/{uid}` Firestore document. The callback
  receives the current `list[str]` of pool IDs every time the user
  document changes, so consumers can detect pool additions or removals
  in the Hayward app without polling `get_pools()` on a timer.
- `AquariteClient.subscribe_user_pools_resilient(callback, *, ...)` —
  same payload, wrapped with the existing supervisor (token-refresh
  resubscribe, exponential backoff, idempotent `aclose()`). Returns a
  `ResilientUserPoolsSubscription` handle.
- `ResilientUserPoolsSubscription`, exported from `aioaquarite`.

### Changed
- The supervisor logic in `ResilientPoolSubscription` is now lifted
  into a private `_ResilientSubscription` base class so the new
  user-pools variant doesn't duplicate ~50 lines of reconnect/refresh
  bookkeeping. The public surface of `ResilientPoolSubscription` is
  unchanged (constructor signature, `pool_id` property, `aclose()`
  semantics, log behaviour).

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
