"""Processor: ingestion, KDE compression, Perfetto export (§5)."""

from argus.processor.compress import compress_kernel_events, kde_valley_clusters
from argus.processor.pipeline import Processor

__all__ = ["Processor", "compress_kernel_events", "kde_valley_clusters"]
