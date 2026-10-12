"""
build123d kernel facade

The topology layer reaches the modeling kernel only through ``kernel``, an
instance of :class:`~build123d.kernel.base.Kernel`. OpenCascade is the only
backend today; see ``base.py`` for the interface.
"""

from build123d.kernel.base import BoolOp, Handle, Kernel, TopoKind
from build123d.kernel.occt import OcctKernel

kernel: Kernel = OcctKernel()

__all__ = ["BoolOp", "Handle", "Kernel", "TopoKind", "OcctKernel", "kernel"]
