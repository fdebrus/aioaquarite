"""Async Python client for the Hayward Aquarite pool API."""

from google.cloud.firestore_v1.watch import Watch

from .auth import AquariteAuth
from .client import AquariteClient
from .exceptions import AquariteError, AuthenticationError, CommandError, ConnectionError
from .subscription import ResilientPoolSubscription, ResilientUserPoolsSubscription

__all__ = [
    "AquariteAuth",
    "AquariteClient",
    "AquariteError",
    "AuthenticationError",
    "CommandError",
    "ConnectionError",
    "ResilientPoolSubscription",
    "ResilientUserPoolsSubscription",
    "Watch",
]
