"""Closed-form antenna design equations (seed geometries and benchmarks)."""

from __future__ import annotations

import math

from gradenna.constants import C0


def patch_design(fr: float, eps_r: float, h: float) -> tuple[float, float, float]:
    """Balanis transmission-line design of a rectangular microstrip patch.

    Args:
        fr: design (resonant) frequency [Hz].
        eps_r: substrate relative permittivity.
        h: substrate height [m].

    Returns:
        (W, L, eps_reff): patch width and length [m] and the effective
        permittivity, per Balanis 4th ed. Eqs. (14-6), (14-1), (14-2)
        (Hammerstad fringing extension) and (14-7).
    """
    w = C0 / (2.0 * fr) * math.sqrt(2.0 / (eps_r + 1.0))
    eps_reff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / w) ** -0.5
    dl = (
        0.412
        * h
        * ((eps_reff + 0.3) * (w / h + 0.264))
        / ((eps_reff - 0.258) * (w / h + 0.8))
    )
    length = C0 / (2.0 * fr * math.sqrt(eps_reff)) - 2.0 * dl
    return w, length, eps_reff
