"""
Problem loaders for importing routing instances.

This package provides loaders for accessing routing problem instances
from various sources. Currently supports DuckDB database loading.

Exports:
    DatabaseLoader: Context manager for loading from DuckDB database
"""

from .database_loader import DatabaseLoader

__all__ = ["DatabaseLoader"]
