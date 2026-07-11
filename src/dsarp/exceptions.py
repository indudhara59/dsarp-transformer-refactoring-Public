"""Explicit domain exceptions."""


class DsarpError(RuntimeError):
    """Base exception for expected pipeline failures."""


class InputFileError(DsarpError):
    """Required input is missing or ambiguous."""


class SchemaError(DsarpError):
    """An input schema cannot be mapped safely."""


class JoinError(DsarpError):
    """A join violates identity or cardinality constraints."""


class ConfigurationError(DsarpError):
    """A configuration file is malformed."""

