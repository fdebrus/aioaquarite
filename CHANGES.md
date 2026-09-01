# Changelog

## 0.12.0

### Changed
- **The real-time listener is now natively async.** `subscribe_pool` and
  `subscribe_user_pools` no longer run the synchronous Firestore `Watch`
  through `asyncio.to_thread`; each subscription is an `AsyncDocumentWatch`
  speaking the `Listen` RPC directly on the async gRPC layer
  (`FirestoreAsyncClient.listen`), as an asyncio task on the running loop.
  The library now contains no `asyncio.to_thread` calls and creates no
  threads.
- **Snapshot callbacks are invoked on the event loop**, not on a Firestore
  background thread. `loop.call_soon_threadsafe` is no longer needed in
  consumers (existing code using it keeps working — it just adds a hop).
- `subscribe_pool` / `subscribe_user_pools` now return only after the
  server confirms a first consistent snapshot (CURRENT), so a subscribe
  that cannot deliver data fails loudly instead of returning a silently
  dead watch. The returned watch still has synchronous `unsubscribe()` and
  `await`-able `aclose()`.
- The resilient supervisors now notice a dead stream the moment its task
  finishes (health transition + backoff + resubscribe), instead of only at
  the next health-check tick. Reconnects resume from the last consistent
  point via the server's resume token rather than replaying from scratch.

### Unchanged
- `subscribe_pool_resilient` / `subscribe_user_pools_resilient` signatures,
  `on_health` transition-only semantics, `aclose()`, and
  `AquariteAuth.get_client()`'s `tuple[Client, int]` contract are all
  exactly as in 0.11.0.
- `from aioaquarite import Watch` still works as a vestigial re-export of
  the upstream class; new code should use `AsyncDocumentWatch`.

## 0.11.0

### Fixed
- Every subscription on an account now resubscribes when the shared token
  rotates. `_ensure_fresh_clients()` reported a rotation as a per-call
  boolean, set only for the caller that actually ran the refresh under the
  lock. Every other supervisor read `False`, took neither the "token
  refreshed" nor the "watch not established" branch, and kept a watch bound
  to credentials expiring 300 seconds later. That watch then died silently
  at expiry — no exception, no health transition — so consumers kept their
  entities available and serving frozen values until some later rotation
  happened to be won by that supervisor. Severity scaled with pool count:
  N pools meant N+1 supervisors competing for a single boolean.

### Changed
- **`get_client()` now returns `tuple[Client, int]`.** The second element is
  a token generation counter, bumped whenever the Firestore clients are
  replaced, in place of the previous "a refresh just happened" boolean. A
  counter can be read by every caller; the boolean was consumed by the first
  one. Callers holding a long-lived listener should record the generation
  they subscribed with and resubscribe whenever it differs. Testing the
  value for truthiness — valid under the boolean contract — now resubscribes
  on every check instead.

### Added
- `AquariteAuth.token_generation` — the same counter as a property, for
  callers that need to read it without requesting a client.

## 0.10.0

### Changed
- Firestore document reads are now natively async. `get_pools()` and
  `fetch_pool_data()` await `AsyncClient` document references instead of
  dispatching the synchronous client through `asyncio.to_thread`. The
  library previously exposed coroutines while running every Firestore
  operation in a thread pool; reads and commands are now genuinely
  non-blocking.
- `AquariteAuth` builds an `AsyncClient` alongside the existing
  synchronous `Client` from the same credentials, and rotates and closes
  both together. Previously only the synchronous client was closed on
  rotation.

### Added
- `AquariteAuth.get_async_client()` — accessor for the async Firestore
  client used by the read paths. `get_client()` keeps its
  `tuple[Client, bool]` contract for the subscription path, where the
  boolean still signals "token refreshed, resubscribe".
- `AquariteAuth.close()` — releases both Firestore clients. The aiohttp
  session is caller-owned and deliberately untouched.

### Known limitation
- The real-time listener still runs the synchronous client in a thread.
  `google-cloud-firestore` implements `on_snapshot` only on the
  synchronous client; `AsyncDocumentReference.on_snapshot` raises
  `NotImplementedError` (verified against 2.29.0). `subscribe_pool()`,
  `subscribe_user_pools()` and the subscription teardown therefore keep
  their `asyncio.to_thread` calls, now documented in place. A test pins
  the upstream behaviour so the split can be revisited if it changes.

The public API is unchanged.

## 0.9.2

### Fixed
- `filtration.timerVel1/2/3` are now coerced to `int`, not `bool`. They
  carry a three-state pump speed (`0` slow, `1` medium, `2` high) — the
  same semantics as `filtration.manVel`, which was already typed `int`.
  Typed as `bool`, `get_value()` returned the caller's default (`None`)
  for **high** and logged a warning on every read, so a timer interval
  set to high speed surfaced as `unknown` in consumers and spammed the
  log on each Firestore push. Slow and medium happened to survive
  because `bool` round-trips back to index `0`/`1`. String-encoded
  values (`"2"`, sent by some firmware revisions) were affected too.
  Writes were never affected — outgoing values are not coerced.
  Reported by @ThierryR42 (#17).

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
