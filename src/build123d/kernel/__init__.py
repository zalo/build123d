"""
build123d kernel facade

The topology layer reaches the modeling kernel only through ``kernel``, an
instance of :class:`~build123d.kernel.base.Kernel`. OpenCascade is the only
backend today; see ``base.py`` for the interface.
"""

import os

from build123d.kernel.base import (
    BoolOp,
    CurveView,
    Handle,
    JoinKind,
    Kernel,
    SurfaceView,
    SweepTransition,
    TopoKind,
    WireStatus,
)
from build123d.kernel.base import AncestorMap, DraftFailure, SurfaceFilling
from build123d.kernel.occt import OcctKernel


def _select_backend() -> Kernel:
    """The backend named by ``BUILD123D_KERNEL``: ``occt`` (default), ``smlib``
    (OCCT handles, solid operations through usd_brep) or ``smcore`` (SMLib
    handles through the generated kernel-level bindings)."""
    name = os.environ.get("BUILD123D_KERNEL", "occt").strip().lower()
    if name == "occt":
        return OcctKernel()
    if name == "smlib":
        from build123d.kernel.smlib import (
            SmlibKernel,
        )  # pylint: disable=import-outside-toplevel

        return SmlibKernel()
    if name == "smcore":
        from build123d.kernel.smlib_core import (  # pylint: disable=import-outside-toplevel
            SmlibCoreKernel,
        )

        return SmlibCoreKernel()
    raise ValueError(
        f"unknown kernel backend {name!r}; use 'occt', 'smlib' or 'smcore'"
    )


kernel: Kernel = _select_backend()

__all__ = [
    "AncestorMap",
    "BoolOp",
    "DraftFailure",
    "SurfaceFilling",
    "CurveView",
    "Handle",
    "JoinKind",
    "Kernel",
    "SurfaceView",
    "SweepTransition",
    "TopoKind",
    "WireStatus",
    "OcctKernel",
    "kernel",
]
