"""Async Python client for the Hayward Aquarite pool API."""

# Vestigial re-export: subscribe_* returned this class before 0.12.0.
# Kept so `from aioaquarite import Watch` does not break; new code should
# use AsyncDocumentWatch.
from google.cloud.firestore_v1.watch import Watch

from ._watch import AsyncDocumentWatch
from .auth import AquariteAuth
from .client import AquariteClient
from .exceptions import AquariteError, AuthenticationError, CommandError, ConnectionError
from .subscription import ResilientPoolSubscription, ResilientUserPoolsSubscription

__all__ = [
    "AquariteAuth",
    "AquariteClient",
    "AquariteError",
    "AsyncDocumentWatch",
    "AuthenticationError",
    "CommandError",
    "ConnectionError",
    "ResilientPoolSubscription",
    "ResilientUserPoolsSubscription",
    "Watch",
]
