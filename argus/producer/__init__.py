"""Trace Producer (§4): CPU stacks, framework semantics, GPU kernels."""

from argus.producer.agent import TraceProducer
from argus.producer.semantics import SemanticsTracer

__all__ = ["TraceProducer", "SemanticsTracer"]
