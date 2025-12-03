from typing import Callable, Dict

from .base import RecModelBase

MODEL_REGISTRY: Dict[str, Callable[[dict], RecModelBase]] = {}


def register_model(name: str):
    """
    Decorator to register a model builder.
    Builder signature: fn(model_cfg: dict) -> RecModelBase
    """

    def decorator(fn: Callable[[dict], RecModelBase]):
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def build_model(model_cfg: dict) -> RecModelBase:
    name = model_cfg.get("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Registered: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](model_cfg)
