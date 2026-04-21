"""Shared types and enums for the attack framework."""

from enum import Enum


class PrivacyMode(Enum):
    """Privacy mode controlling what victim data is written to Sacred.

    FULL_WHITEBOX: Victims write Q-tables, env dynamics, and behavior traces.
    FULL_BLACKBOX: Victims write only behavior traces.
    """
    FULL_WHITEBOX = 0
    FULL_BLACKBOX = 1
