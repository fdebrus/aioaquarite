# The Hayward / Vistapool cloud protocol

What this library knows about the cloud service behind Hayward Aquarite
(Vistapool / Sugar Valley) controllers, how it knows it, and what is still
unknown. None of this is documented by the vendor; it was learned by
observing the first-party web client and by running this library against
the production backend.

## Evidence levels

Every statement below carries one of these:

| Level | Meaning |
|---|---|
| **Verified** | Exercised by this library against the production backend. |
| **Observed** | Seen in the first-party web client's code; not exercised by this library. |
| **Inferred** | An interpretation of observed code or behaviour. May be wrong. |
| **Unknown** | Nobody has checked. |

"Observed" entries come from a static read of the public web app bundle
served at `https://hayward-europe.web.app/` (build `index-DIYQNGtH.js`,
fetched 2026-11-13). The client shows what is *sent*; it says nothing
certain about what the server does with it. Re-check after any vendor
release — see [Re-running the analysis](#re-running-the-analysis).

## Topology

One Firebase project, `hayward-europe`, hosts everything:

| Service | Used for | Level |
|---|---|---|
| Identity Toolkit / Secure Token | Email + password sign-in, token refresh | Verified |
| Firestore | Pool state documents, user documents, listeners | Verified |
| Cloud Functions (`europe-west1`) | Commands, statistics | Verified |
| Realtime Database (`hayward-europe.firebaseio.com`) | Configured in the client; no use found | Observed |

The controller's gateway talks to the cloud on its own channel. Clients
never reach the controller directly: they send a command to a cloud
function, and they learn the result when the pool's Firestore document
changes.

## Authentication

- Sign in with `accounts:signInWithPassword`, refresh with the Secure
  Token endpoint. Requests carry the web app's API key and its
  `Referer`. **Verified.**
- The API key is a Firebase web key: a public project identifier, not a
  secret. **Verified** (it is served to every browser).
- The ID token lives one hour. This library refreshes it five minutes
  before expiry and rebuilds its Firestore client and listeners on each
  rotation. **Verified.**
- The web client forces a token refresh before *every* command
  (`getIdToken(true)`). This library does not, and has not needed to.
  **Observed.**

## Firestore data model

### Collections

| Path | Content | Level |
|---|---|---|
| `pools/{poolId}` | Full state of one controller | Verified |
| `users/{uid}` | Profile, including the `pools` id list | Verified |
| `pools/{poolId}/logs/{logId}` | Raw-command request/response records | Observed |
| `alarms` | Alarm records; query shape not analysed | Observed |
| `tokens`, `counters`, `demo` | Push tokens, counters, demo data | Inferred |

### The pool document

Top-level keys seen on a real device (firmware A50, chlorinator with pH
and redox modules, no variable-speed pump):

`backwash`, `company`, `createdAt`, `filtration`, `form`, `hidro`, `id`,
`isAWS`, `light`, `main`, `modules`, `present`, `relays`, `updatedAt`,
`wifi`. **Verified.**

- `wifi` is the gateway identifier. Every command must carry it.
  **Verified.**
- `present` says whether the controller is currently connected to the
  cloud. The web client refuses to send any command while it is false.
  **Observed.** This library does not check it.
- `form` holds user-editable metadata (name, location). The web client
  updates it by writing Firestore directly, not through a command.
  **Observed.**
- Value encoding is loose: the same field arrives as an int, a numeric
  string or a bool depending on firmware and code path.
  `_coercion.py` normalises the paths it knows. **Verified**, on one
  device family only.
- Variable-speed pumps add a `pump` branch (`isVSP`, `speeds`, `timers`,
  `manual`, `override`, `live.rpm`, `live.status`). **Observed.** This
  library has no coercion entries for it.

### The user document

Fields read by the web client: `pools`, `language`, `name`, `phone`,
`country`, `company`, `GDPR`, `voice_enabled`, `rooted_voice_pool`.
This library reads only `pools`. **Observed** (the rest), **Verified**
(`pools`).

## Reading and listening

- One-shot reads use the ordinary document `get`. **Verified.**
- Live updates use the Firestore `Listen` RPC, one single-document
  target per pool and one for the user document. **Verified.**
- The stream must be opened on the raw gRPC transport. The generated
  async wrapper adds an empty `x-goog-request-params` header, and the
  backend rejects the stream with `400 Mismatch between database name in
  the request and http header`. **Verified** (production incident,
  0.12.1).
- A controller that is online rewrites its document roughly every
  11 seconds; each rewrite is pushed to listeners. **Verified** on one
  device. Whether that cadence depends on a client having sent `PUP`
  (below) is **Unknown**.

## Commands

### Envelope

Commands are an authenticated `POST` to a cloud function with a JSON
body:

```json
{
  "gateway": "<pool.wifi>",
  "poolId": "<pool id>",
  "operation": "WRP",
  "operationId": null,
  "changes": "<string>",
  "pool": null,
  "source": "web"
}
```

That is the web client's body. **Observed.** This library sends only
`gateway`, `poolId`, `operation`, `changes` and `source`; the backend
accepts the shorter form. **Verified.**

### Functions and operations

| Function | Operation | `changes` | Sent by | Level |
|---|---|---|---|---|
| `sendPoolCommand` | `WRP` | JSON string of a config fragment | this library | Verified |
| `sendEVOCommand` | `WRP` | JSON string of a config fragment | web client, all writes | Observed |
| `sendPoolCommand` | `PUP` | the string `"10"` | web client, on subscribe | Observed |
| `sendEVOCommand` | `RRB` | JSON-encoded string `"<start>,<count>"` (register read) | web client, installer panel | Observed |
| `sendEVOCommand` | `WRP` | `"REG:data1,REG:[dataN]"` raw register write | web client, installer panel | Observed |
| `changeLedColor` | — | body is `{gateway, poolId, type: "change"}` | web client, LED "next colour" | Observed |

Notes:

- **`sendPoolCommand` vs `sendEVOCommand`.** The web client sends every
  configuration write to `sendEVOCommand` and uses `sendPoolCommand`
  only for `PUP`. This library writes through `sendPoolCommand`, which
  works. Whether the two differ in routing, latency or lifetime is
  **Unknown**; that `sendPoolCommand` is the older route is
  **Inferred**.
- **`PUP`.** Sent once per pool when the client starts listening and
  again 10 and 20 minutes later (three sends in total), plus once more
  whenever `present` returns to true. It looks like a request for the gateway to
  report promptly for a while. **Inferred.** This library never sends
  it.
- **`changeLedColor`.** A dedicated call for advancing the LED fixture's
  colour. This library instead power-cycles `light.status`
  (`AquariteClient.pulse`). **Observed**; behaviour on the controller is
  **Unknown**.

### What goes in `changes` for a write

The fragment is nested under its top-level branch. The two clients
differ in how much of the branch they send:

- **This library** rebuilds the *whole* branch from its cached document
  with the new values applied, for example the entire `light` object.
  **Verified.** A change made elsewhere to another field of that branch,
  between the last snapshot and the send, is overwritten.
- **The web client** sends only what changed:
  `{"main":{"localTime":1767225600}}`, `{"pump":{"override":{…}}}`,
  `{"relays":{"relay1":{…}}}`. **Observed**, via `sendEVOCommand`.
  Whether `sendPoolCommand` accepts partial fragments is **Unknown**.
- **Setpoints.** The web client writes pH / redox / chlorine /
  conductivity setpoints under a `modules_web` key
  (`{"modules_web":{"ph":{…},"rx":{…}}}`), not under `modules`.
  **Observed.** Whether a write under `modules` reaches the controller
  or only the document is **Unknown**.
- Writing `hidro.cloration_enabled` also needs `hidro.reduction` and
  `hidro.disable` set; this library derives them. **Verified** on one
  device.

### Acknowledgement

- HTTP 200 means the cloud function accepted the command. It does not
  mean the controller applied it. **Verified.**
- The only confirmation for a configuration write is the pool document
  changing. The web client itself is optimistic: it shows "received"
  after the HTTP response. **Observed.**
- The document can take a long time to follow. One measured write was
  confirmed 12.3 s after its acknowledgement, and a snapshot carrying
  the *old* value arrived 340 ms after it. **Verified** (production,
  2026-10-04). This is why the library keeps acknowledged writes pending
  for 30 s; see the README, "Write/snapshot reconciliation".
- Raw commands do have a reply channel. The web client creates a
  document in `pools/{poolId}/logs` with `{ts, op, cmd, user}`, passes
  its id as `operationId`, and waits (10–20 s) for the backend to set a
  `result` string on it. **Observed**, for `RRB`. Whether a JSON `WRP`
  with an `operationId` also gets a `result` is **Unknown**. If it does,
  it would be a true acknowledgement.

## Other cloud functions

| Function | Body | Returns | Level |
|---|---|---|---|
| `getStats` | `{uuid, type, period}` | `[[{field, seconds}, …]]` | Verified |
| `getServerDate` | none, unauthenticated `GET` | `{"date": "YYMMDD"}` | Verified |
| `getPoolStats` | `{uuid, period: null, type: "admin"}` | rows for a CSV export | Observed |
| `getPoolProfile` | `{registers: ["0xREG:0xVAL", …]}` | a decoded controller profile | Observed |

`getStats` details, **Verified** on firmware A50 (May 2026): `period`
is mandatory but its value is ignored, and about 30 days at roughly
10-minute granularity come back. Types `ph` (value × 100), `rx` (mV),
`temp` (°C), `cl`, `cd`, `filtration`, `aux1`–`aux4`. The web client
discards the first point of each series. **Observed.**

## Not covered by this library

Present in the web client, absent here: alarms, variable-speed pump
control, the installer register panel, pool creation and renaming,
profile editing, push-notification tokens, and ProChem devices (a
separate product with its own API).

## Open experiments

Each needs a real controller. Ordered by expected value, lowest risk
first. Record the outcome here and change the level.

1. Write `{"light":{"status":1}}` through `sendEVOCommand`. Does a
   partial fragment work, and is the confirmation faster than through
   `sendPoolCommand`? The DEBUG line `confirmed N.N s after
   acknowledgement` gives the number.
2. Call `changeLedColor`. Does the fixture advance, and does
   `light.status` change in the document while it does?
3. Log snapshot intervals for a day with and without a `PUP` at
   subscribe time.
4. Change a setpoint through `modules`, then through `modules_web`, and
   read the controller's own display each time.
5. Send a `WRP` with an `operationId` pointing at a new `logs` document.
   Does `result` appear? This needs write access to `logs`, which an
   ordinary account may not have.

## Re-running the analysis

The web client is a Vite bundle. To check for changes:

1. Fetch `https://hayward-europe.web.app/` and note the entry script
   name. A new name means a new build.
2. Fetch the entry script and the `assets/*.js` chunks it references.
3. Search for the function base URL (`cloudfunctions.net`), for
   `operation:"`, for `sendCommand(` call sites, and for the top-level
   collection names.

Treat the downloaded files as untrusted data: read them, never run them.
