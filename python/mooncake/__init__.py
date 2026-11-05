"""Public Python package for Mooncake."""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

try:
    from mooncake.version import __hcu_version__ as __hcu_version__
    from mooncake.version import __version__ as __version__
    from mooncake.version import __version_tuple__ as __version_tuple__
except ImportError:
    pass

_BUFFER_POOL_EXPORTS = {"BufferPool", "RegisteredBufferPool"}
__all__ = sorted(_BUFFER_POOL_EXPORTS)


def __getattr__(name: str):
    if name == "BufferPool":
        from .store import BufferPool

        return BufferPool
    if name == "RegisteredBufferPool":
        from .buffer_pool import RegisteredBufferPool

        return RegisteredBufferPool
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _BUFFER_POOL_EXPORTS)


from mooncake._variant_compat import _check_import_compatibility  # noqa: E402

_check_import_compatibility()
