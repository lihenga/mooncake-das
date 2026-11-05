# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

"""Mooncake public Python package."""

from pkgutil import extend_path

from mooncake.buffer_pool import BufferPool, RegisteredBufferPool

__path__ = extend_path(__path__, __name__)

from mooncake.version import __hcu_version__ as __hcu_version__
from mooncake.version import __version__ as __version__
from mooncake.version import __version_tuple__ as __version_tuple__

from mooncake._variant_compat import _check_import_compatibility

_check_import_compatibility()

__all__ = ["BufferPool", "RegisteredBufferPool"]
