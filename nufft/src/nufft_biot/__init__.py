from .types import BoxParams

__all__ = ["BoxParams", "forward_B"]


def __getattr__(name):
    if name == "forward_B":
        from .forward import forward_B

        return forward_B
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
