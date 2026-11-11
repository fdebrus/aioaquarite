"""Async Python client for the Hayward Aquarite pool API."""

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
]
