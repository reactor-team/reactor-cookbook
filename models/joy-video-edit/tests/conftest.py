# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Fixtures for the joy-video-edit tests.

The tests drive the application half through its hooks and handlers with a
fake model half, and the model half's run bookkeeping with the JoyOmni
runtime replaced by a fake, so they run without a GPU, the weights, or torch.
When torch is absent it is stubbed; inside the model image it imports for
real and the same tests run. The vendored ``xvideo`` package is imported only
inside ``load()``, which the tests never call.

Run from the model directory::

    PYTHONPATH=. python -m pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest

_MODEL_DIR = Path(__file__).resolve().parent.parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))


class _Stub:
    """Stands in for any attribute of a stubbed module: callable, iterable, a context manager."""

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    def __call__(self, *args: object, **kwargs: object) -> _Stub:
        return _Stub()

    def __getattr__(self, name: str) -> _Stub:
        return _Stub()

    def __iter__(self):
        return iter(())

    def __enter__(self) -> None: ...

    def __exit__(self, *exc: object) -> None: ...


class _StubModule(types.ModuleType):
    def __getattr__(self, name: str) -> _Stub:
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub()


if importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", _StubModule("torch"))


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run an ``async def`` test on a fresh event loop, as the app's hooks are coroutines."""
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    arguments = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(pyfuncitem.obj(**arguments))
    return True
