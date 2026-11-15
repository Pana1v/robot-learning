"""BARN navigation core: pure-Python, no Isaac Lab / Isaac Sim imports.

Testable without Kit. Holds the offline world parser, the navigation metric
and geometry primitives shared by the Isaac Lab env and the eval pipeline.

Upstream: Daffan/the-barn-challenge@bf5a226
"""

from . import geometry, metric, worlds

__all__ = ["geometry", "metric", "worlds"]
