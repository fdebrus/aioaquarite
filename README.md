# aioaquarite

[![PyPI](https://img.shields.io/pypi/v/aioaquarite)](https://pypi.org/project/aioaquarite/)
[![Python versions](https://img.shields.io/pypi/pyversions/aioaquarite)](https://pypi.org/project/aioaquarite/)
[![Tests](https://github.com/fdebrus/aioaquarite/actions/workflows/tests.yml/badge.svg)](https://github.com/fdebrus/aioaquarite/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

Async Python client for the Hayward Aquarite pool API.

This library provides a standalone API client for interacting with Hayward Aquarite pool equipment via the Hayward cloud service. It is designed to be used as the backend for the [Home Assistant Aquarite integration](https://github.com/fdebrus/hayward-ha).

## Features

- **Auth**: email/password sign-in against Firebase Identity Toolkit, with automatic token refresh.
- **Read**: list pools, fetch full pool documents, read individual fields with type coercion.
- **Write**: atomic single- or multi-field commands (`set_value` / `set_values`), with the local cache kept in sync so back-to-back writes never revert each other.
- **Real-time**: resilient Firestore subscriptions (pool data and the user's pool list) with automatic token-refresh reconnects and exponential backoff.
- **History**: pull stored sample series (pH, ORP, temperature, filtration, aux relays, …) and check clock drift against the Hayward backend.
- **Typed errors**: every failure mode raises an `AquariteError` subclass, so callers only need one `except` clause.

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

## Real-time updates

Subscribe with built-in token refresh and automatic reconnect (recommended).
Callbacks run on the Firestore background thread — asyncio consumers should
wrap them with `loop.call_soon_threadsafe`.

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

### Low-level subscriptions

If you want to own the connection lifecycle yourself, the raw watch handles
are still available for both pool data and the user's pool list:

```python
watch = await client.subscribe_pool(pool_id, on_pool_update)
# ... maintain token freshness, resubscribe on errors, etc. ...
watch.unsubscribe()

watch = await client.subscribe_user_pools(on_pools_changed)
watch.unsubscribe()
```

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
