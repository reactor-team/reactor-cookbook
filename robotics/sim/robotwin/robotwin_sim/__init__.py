"""xwam RoboTwin 2.0 gateway example.

See README.md. This example is a gateway between the RoboTwin 2.0 authors'
evaluation client and a Reactor-served policy, not an env wrapper. Their
client runs unmodified; only transport and serving are substituted.
"""

__all__ = ["contract", "tracks", "bridge", "gateway", "main"]
