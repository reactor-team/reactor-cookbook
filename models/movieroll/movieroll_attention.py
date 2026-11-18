"""Explicit FA4 compatibility for MovieRoll's hard-imported FA3 interface."""


def configure() -> None:
    import sys
    from types import ModuleType

    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

    if "movieroll.models.transformers.modules.streaming_attention" in sys.modules:
        raise RuntimeError("Select attention before importing MovieRoll")
    # The native module imports this interface even for non-Hopper execution.
    # Preserve its dense/varlen arguments and masks, using FA4 on Blackwell.
    # Different kernels can round differently; this is not a bitwise FA3 claim.
    interface = ModuleType("flash_attn_interface")
    interface.flash_attn_func = flash_attn_func
    interface.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn_interface"] = interface
