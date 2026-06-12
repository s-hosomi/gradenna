"""Yee grid definitions for the 2D TM and 3D solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from gradenna.constants import C0


@dataclass(frozen=True)
class Grid2D:
    """Uniform 2D Yee grid.

    Field locations follow the standard Yee staggering:
    Ez at integer points (i, j), Hx at (i, j+1/2), Hy at (i+1/2, j).
    The outermost Ez ring is held at zero (PEC outer boundary).
    """

    nx: int
    ny: int
    dx: float
    dy: float
    courant: float = 0.99  # fraction of the 2D stability limit

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError(f"grid must be at least 3x3, got {self.nx}x{self.ny}")
        if self.dx <= 0.0 or self.dy <= 0.0 or self.courant <= 0.0:
            raise ValueError("dx, dy and courant must be positive")

    @property
    def dt(self) -> float:
        """Time step Δt = S / (c √(1/Δx² + 1/Δy²)) with S = courant."""
        return self.courant / (C0 * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)


@dataclass(frozen=True)
class Grid3D:
    """Uniform 3D Yee grid (see the gradenna.fdtd3d module docstring for
    the field locations)."""

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    courant: float = 0.99  # fraction of the 3D stability limit

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) < 3:
            raise ValueError(
                f"grid must be at least 3x3x3, got {self.nx}x{self.ny}x{self.nz}"
            )
        if min(self.dx, self.dy, self.dz) <= 0.0 or self.courant <= 0.0:
            raise ValueError("dx, dy, dz and courant must be positive")

    @property
    def dt(self) -> float:
        """Time step Δt = S / (c √(1/Δx² + 1/Δy² + 1/Δz²)) with S = courant."""
        return self.courant / (
            C0 * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2 + 1.0 / self.dz**2)
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)
