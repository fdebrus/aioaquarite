# aioaquarite

Async Python client for the Hayward Aquarite pool API.

This library provides a standalone API client for interacting with Hayward Aquarite pool equipment via the Hayward cloud service. It is designed to be used as the backend for the [Home Assistant Aquarite integration](https://github.com/fdebrus/hayward-ha).

## Installation

```bash
pip install aioaquarite
```

## Usage

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

    # Subscribe with built-in token refresh + reconnect (recommended).
    # The callback is invoked from the Firestore background thread;
    # asyncio consumers should wrap it with loop.call_soon_threadsafe.
    def on_update(data):
        print("Pool updated:", data.get("main", {}).get("temperature"))

    subscription = await client.subscribe_pool_resilient(pool_id, on_update)

    await client.set_value(pool_id, "filtration.mode", 1)

    # Cleanup
    await subscription.aclose()
```

### Low-level subscription (0.3-style)

If you want to own the connection lifecycle yourself, the raw watch handle is
still available:

```python
watch = await client.subscribe_pool(pool_id, on_update)
# ... maintain token freshness, resubscribe on errors, etc. ...
watch.unsubscribe()
```

## License

MIT
