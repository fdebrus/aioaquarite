# aioaquarite

<p align="left">
  <a href="https://www.buymeacoffee.com/fdebrus"><img src="https://img.shields.io/badge/Support-Buy%20Me%20a%20Coffee-FFDD00?style=flat&logo=buymeacoffee" alt="Buy Me a Coffee"></a>
  <a href="https://pypi.org/project/aioaquarite/"><img src="https://img.shields.io/pypi/v/aioaquarite?style=flat&label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/aioaquarite/"><img src="https://img.shields.io/pypi/pyversions/aioaquarite?style=flat&label=Python" alt="Python versions"></a>
  <a href="https://github.com/fdebrus/aioaquarite/actions/workflows/tests.yml"><img src="https://github.com/fdebrus/aioaquarite/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/fdebrus/aioaquarite"><img src="https://img.shields.io/badge/Maintained%20by-fdebrus-green?style=flat" alt="Maintainer"></a>
  <a href="https://github.com/fdebrus/aioaquarite/issues"><img src="https://img.shields.io/github/issues/fdebrus/aioaquarite?style=flat&label=Issues" alt="Open issues"></a>
  <a href="https://github.com/fdebrus/aioaquarite/stargazers"><img src="https://img.shields.io/github/stars/fdebrus/aioaquarite?style=flat&label=Stars" alt="GitHub stars"></a>
</p>

Async Python client for the Hayward Aquarite pool API.

This library provides a standalone API client for interacting with Hayward Aquarite pool equipment via the Hayward cloud service. It is designed to be used as the backend for the [Home Assistant Aquarite integration](https://github.com/fdebrus/hayward-ha).

## Features

- **Auth**: email/password sign-in against Firebase Identity Toolkit, with automatic token refresh.
- **Read**: list pools, fetch full pool documents, read individual fields with type coercion.
- **Write**: atomic single- or multi-field commands (`set_value` / `set_values`), with the local cache kept in sync so back-to-back writes never revert each other.
- **Real-time**: resilient Firestore subscriptions (pool data and the user's pool list) with automatic token-refresh reconnects, exponential backoff, and connection-health reporting (`on_health` callback / `healthy` property).
- **History**: pull stored sample series (pH, ORP, temperature, filtration, aux relays, …) and check clock drift against the Hayward backend.
- **Typed errors**: every failure mode raises an `AquariteError` subclass, so callers only need one `except` clause.

## Async model

Document reads (`get_pools`, `fetch_pool_data`) run on the Firestore
`AsyncClient` and are awaited directly. The command and history endpoints
(`set_value`, `set_values`, `send_command`, `get_pool_stats`,
`get_server_date`) are plain `aiohttp` calls. Neither path blocks the event
loop or dispatches to a thread pool.

The **real-time listener is native too** (since 0.12.0):
`google-cloud-firestore` implements `on_snapshot` only on its synchronous
client — `AsyncDocumentReference.on_snapshot` raises `NotImplementedError`
(checked against 2.29.0) — so the library speaks the underlying `Listen`
RPC itself, through the async gRPC layer the package already ships
(`FirestoreAsyncClient.listen`). Each subscription is an asyncio task on
the running loop; nothing in the library creates a thread or calls
`asyncio.to_thread`. **Snapshot callbacks are invoked on the event loop**,
so no `loop.call_soon_threadsafe` is needed anywhere (existing code that
still uses it keeps working — it just adds a needless hop).

Both clients are built from the same credentials, rotated together on token
refresh, and released by `AquariteAuth.close()`.

## Installation

```bash
pip install aioaquarite
```

Requires Python 3.12+.

## Quick start

```python
import aiohttp
from aioaquarite import AquariteAuth, AquariteClient

async with aiohttp.ClientSession() as session:
    auth = AquariteAuth(session, "user@example.com", "password")
    await auth.authenticate()

    # Stable Firebase UID (`sub` claim of the id token); useful as a
    # config-entry unique_id. Returns None before authenticate() succeeds.
    print("Firebase UID:", auth.user_id)

    client = AquariteClient(auth)

    pools = await client.get_pools()
    for pool_id, pool_name in pools.items():
        data = await client.fetch_pool_data(pool_id)
        temperature = AquariteClient.get_value(data, "main.temperature")
        print(f"{pool_name}: {temperature}°C")
```

## Writing values

`set_value` writes a single field; `set_values` writes several fields of the
same command branch as one atomic command — useful when two fields must land
together (e.g. a light's mode and status):

```python
# Single field.
await client.set_value(pool_id, "filtration.mode", 1)

# Several fields, one command — sent together or not at all.
await client.set_values(pool_id, {"light.mode": 2, "light.status": 1})
```

Paths use dot notation (`"hidro.cloration_enabled"`, `"relays.relay1.info.onoff"`).
All paths passed to `set_values` must resolve to the same command branch —
the same top-level key, and for deep 4+ segment paths, the same second-level
key too — mixing branches raises `ValueError`. On a successful send, both
methods immediately update the client's local pool-data cache, so the next
command is built from the state the cloud just acknowledged rather than a
stale Firestore snapshot.

## Write/snapshot reconciliation

Hayward's cloud acknowledges a REST command (HTTP 200) 5–10 seconds before
the Firestore document reflects it. A snapshot emitted in that window
genuinely carries the pre-write state, and no timestamp or ordering can tell
such an echo from a real external change (someone pressing the controller).
Since 0.13.0 the library reconciles the two sides itself, so consumers never
see a toggle flicker back:

- **Every acknowledged write is queued as pending**, per `(pool, path)`,
  with its own timestamp, and **delivered immediately**: the subscriber
  callback fires with the overlaid data the moment the cloud acks the
  command, so consumers reflect a write without waiting out the 5–10 s
  Firestore echo. `set_value` / `set_values` do all of this automatically —
  no consumer code needed.
- **Snapshots confirm pending writes in order.** The queue head is popped
  when the snapshot agrees with it (tolerantly — Firestore returns
  int/str/bool variants); a later write is never confirmed while an earlier
  one is pending, so a pre-write echo cannot lift protection.
- **Delivered data carries the newest pending value** for each protected
  path (the snapshot is overlaid), on the resilient subscriptions, the
  low-level `subscribe_pool`, and `fetch_pool_data` alike.
- **Each write ages out on its own timestamp** (TTL 10 s —
  `aioaquarite._pending.PENDING_WRITE_TTL_SECONDS`, a module constant by
  design, not a consumer knob). An identical repeated write coalesces into
  the existing entry: an idempotent repeat causes no second document
  transition, so a second entry would wait for a confirmation that never
  comes. A snapshot that just pruned an expired write cannot confirm the
  next one on the same pass — it may itself be a pre-write echo.
- **Expiry without confirmation reconciles with the truth**: the overlay is
  dropped and the last raw snapshot re-delivered immediately (last-known
  truth — consumers keep their entities available, no "unavailable" flap),
  then an authoritative fetch runs and its result is delivered. A failed
  fetch retries with backoff (2 s doubling to 60 s); a new snapshot cancels
  the fetch and the retry. The listen stream itself is alive throughout, so
  `on_health` is never involved.
- **An in-flight `fetch_pool_data` cannot regress newer state**: if a
  snapshot (or reconcile fetch) lands while it is reading, the newer state
  is returned instead of the stale read — on the failure path too.

Everything runs on the event loop (the library has had no threads since
0.12), so the only locks are `asyncio` ones.

### Multi-step write sequences

A sequence like an LED colour pulse (off, wait, on) needs its transient
value protected *before* the first send, so a snapshot landing mid-sequence
is overlaid with the final value instead of the intermediate one. The
built-in helper does the whole dance:

```python
# Power-cycle light.status 1 → 0 → wait 1 s → 1, with no visible off.
await client.pulse(pool_id, "light.status", 0, 1, 1.0)
```

It pre-queues both writes, holds the path's write lock across the sends
(so another writer cannot interleave and break queue-order == wire-order),
restarts the final write's TTL after its own send, and on a failed send
discards the never-acknowledged writes and reconciles with an authoritative
fetch.

For custom sequences the same primitives are public:

```python
async with client.write_lock(pool_id, "light.status"):
    client.record_pending(pool_id, "light.status", 0)   # queue, don't send
    client.record_pending(pool_id, "light.status", 1)
    await client.set_value(pool_id, "light.status", 0)  # claims the queued 0
    await asyncio.sleep(1.0)
    await client.set_value(pool_id, "light.status", 1)  # claims the queued 1
    client.refresh_pending(pool_id, "light.status")     # TTL from this send
```

On failure, `client.discard_pending(pool_id, path)` drops the newest queued
write (expiry re-arms from the surviving tail), and `client.reconcile(pool_id)`
fetches the authoritative state in the background and delivers it.

## Real-time updates

Subscribe with built-in token refresh and automatic reconnect (recommended).
Callbacks are invoked on the event loop — call your asyncio code directly
from them, no `loop.call_soon_threadsafe` needed.

```python
def on_pool_update(data):
    print("Pool updated:", data.get("main", {}).get("temperature"))

def on_pools_changed(pool_ids):
    print("User's pools:", pool_ids)

pool_sub = await client.subscribe_pool_resilient(pool_id, on_pool_update)
pools_sub = await client.subscribe_user_pools_resilient(on_pools_changed)

# ... later ...
await pool_sub.aclose()
await pools_sub.aclose()
```

### Connection health

Both resilient subscriptions accept an optional `on_health` callback that
reports connection-state transitions — `on_health(False)` when the
connection is lost, `on_health(True)` once it is re-established. Useful for
marking entities unavailable in a Home Assistant integration while the
Firestore connection is down:

```python
def on_health(healthy: bool) -> None:
    print("Connection healthy:" if healthy else "Connection LOST:", healthy)

pool_sub = await client.subscribe_pool_resilient(
    pool_id, on_pool_update, on_health=on_health
)
print(pool_sub.healthy)  # current connection state
```

`on_health` fires on transitions only, never for `aclose()`, and — unlike
the data callback — is invoked from the event loop running the supervisor
task, so no `call_soon_threadsafe` is needed. An exception raised by the
callback is logged and never kills the supervisor.

### Low-level subscriptions

If you want to own the connection lifecycle yourself, the raw watch handles
are still available for both pool data and the user's pool list:

```python
watch = await client.subscribe_pool(pool_id, on_pool_update)
# ... maintain token freshness, resubscribe on errors, etc. ...
watch.unsubscribe()          # synchronous: cancels the listen task
# or, to wait for full teardown:
await watch.aclose()

watch = await client.subscribe_user_pools(on_pools_changed)
watch.unsubscribe()
```

Each watch is an `AsyncDocumentWatch` running the Firestore `Listen` stream
as a task on your event loop. `await client.subscribe_pool(...)` returns
only once the server has confirmed a first consistent snapshot (and your
callback has seen it); a stream that ends or errors finishes the watch's
`task`, which is how the resilient supervisor notices and reconnects —
resuming from the last consistent point via the saved resume token.

## Historical stats & server clock

```python
# Stored sample series (~30 days, ~10-minute granularity). Each point is
# {"field": <value>, "seconds": <utc_unix>}. Recognised types: ph, rx, temp,
# cl, cd, filtration, aux1..aux4 (plus hardware-conditional light/production/salt).
series = await client.get_pool_stats(pool_id, "ph", period=30)
print("pH samples:", len(series[0]))

# Clock-drift check against the Hayward backend (unauthenticated endpoint).
server_date = await client.get_server_date()
print("Server date:", server_date["date"])  # "YYMMDD"
```

## Error handling

Every failure raises an `AquariteError` subclass, so a single `except` covers
all of them:

```python
from aioaquarite import AquariteError, AuthenticationError, CommandError, ConnectionError

try:
    await client.set_value(pool_id, "filtration.mode", 1)
except AuthenticationError:
    ...  # bad credentials, or refresh token rejected
except ConnectionError:
    ...  # transport failure or timeout talking to the Hayward cloud
except CommandError:
    ...  # the cloud function accepted the connection but rejected the command
except AquariteError:
    ...  # catch-all for anything else in the library
```

## Development

```bash
git clone https://github.com/fdebrus/aioaquarite
cd aioaquarite
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest
python -m pytest tests/
```

Tests run automatically on every push and pull request via [GitHub Actions](.github/workflows/tests.yml).

## License

MIT
