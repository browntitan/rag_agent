"""Standalone notebook runtime for the demo_notebook deliverable.

This package is intentionally isolated from the production application code.
"""

from .config import NotebookSettings, load_settings
from .providers import ProviderBundle, build_provider_bundle
from .stores import PostgresVectorStore
from .orchestrator import DemoOrchestrator

__all__ = [
    "NotebookSettings",
    "load_settings",
    "ProviderBundle",
    "build_provider_bundle",
    "PostgresVectorStore",
    "DemoOrchestrator",
]
