"""Single source of truth for the openEMS cross-check patch geometry.

Both the gradenna model (``tests/test_openems_refs.py``) and the openEMS
reference generator (``generate_patch_refs.py``) build their mesh from the
constants and helpers defined here, so the two simulations describe the
*same physical antenna*. The numbers mirror the 2.45 GHz FR-4 patch of
``tests/test_patch_antenna.py``; they are duplicated here rather than
imported so that this module
stays a self-contained, dependency-light description that the openEMS side
(which has no gradenna on its PYTHONPATH) can also consume.

All lengths are in metres, frequencies in hertz.

Physical antenna (Balanis transmission-line design, FR-4):
    - design / resonant frequency  F0          = 2.45 GHz
    - substrate relative permittivity EPS_FR4   = 4.3
    - substrate height H_SUB                     = 1.6 mm
    - loss tangent  TAN_D                        = 0.02
    - patch W x L derived from ``patch_design`` (~37.6 x 29.1 mm)
    - probe (coaxial / RVS) feed on the radiating edge, centred along W.

The gradenna-specific discretisation knobs (cell size, pin-layer
compensation, CPML thickness, number of steps) live in
``GradennaMesh`` because they are FDTD-implementation details that the
openEMS model does not share. The openEMS mesh is described independently
by its own resolution arguments in ``generate_patch_refs.py`` and recorded
in the CSV metadata; the two solvers only have to agree on the *physics*
captured by ``PatchGeometry`` and the comparison band ``COMPARE_BAND``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- physical constants (SI) -------------------------------------------------
C0 = 299_792_458.0
EPS0 = 8.854_187_8128e-12

# --- design target (mirrors tests/test_patch_antenna.py) ---------------------
F0 = 2.45e9          # design / resonant frequency [Hz]
EPS_FR4 = 4.3        # substrate relative permittivity
H_SUB = 1.6e-3       # substrate height [m]
TAN_D = 0.02         # FR-4 loss tangent at F0
R_PORT = 50.0        # reference / port impedance [ohm]

# Excitation band for the broadband pulse and the S11/Zin sweep window.
# (tests/test_patch_antenna.py excites 1.5-3.5 GHz and sweeps 1.7-3.2 GHz.)
PULSE_BAND = (1.5e9, 3.5e9)     # (f_min, f_max) for the Gaussian pulse [Hz]
SWEEP_BAND = (1.7e9, 3.2e9)     # Zin(f)/S11(f) sweep window [Hz]
N_SWEEP = 301                   # sweep samples across SWEEP_BAND

# Comparison band for the CI test: only inside this window do we trust the
# S11/Zin curves enough to compare RMS errors (both solvers radiate and the
# pulse has energy here). It is the sweep band by default.
COMPARE_BAND = SWEEP_BAND


def patch_design(fr: float, eps_r: float, h: float) -> tuple[float, float, float]:
    """Balanis transmission-line design of a rectangular microstrip patch.

    Standalone copy of :func:`gradenna.designs.patch_design` (Balanis 4th ed.
    Eqs. 14-6, 14-1, 14-2 Hammerstad, 14-7) so the openEMS side, which runs
    without gradenna installed, derives the *identical* W, L. Verified by
    ``tests/test_openems_refs.py`` against the gradenna implementation.

    Returns (W, L, eps_reff) in metres / dimensionless.
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


@dataclass(frozen=True)
class PatchGeometry:
    """Solver-independent description of the benchmark patch antenna.

    This is the contract shared by gradenna and openEMS. Coordinates use a
    patch lying in the z = ``z_patch`` plane (normal +z), substrate filling
    ``0 <= z <= h_sub`` with the ground plane at z = 0. The patch is centred
    on the origin in x (width W along x) and offset in y so its near radiating
    edge is the feed edge (consistent with the openEMS Simple_Patch_Antenna.py
    layout, where the feed sits a small inset from the patch edge).
    """

    f0: float = F0
    eps_r: float = EPS_FR4
    h_sub: float = H_SUB
    tan_d: float = TAN_D
    z0: float = R_PORT

    # Lateral extent of the (finite) substrate / ground plane, expressed as a
    # margin added on each side of the patch footprint. ~6 h matches the
    # M_GND = 8 cells (= 6 h) ground margin of test_patch_antenna.py.
    ground_margin: float = 6.0 * H_SUB

    @property
    def patch_w(self) -> float:
        """Patch width W (along x) [m]."""
        return patch_design(self.f0, self.eps_r, self.h_sub)[0]

    @property
    def patch_l(self) -> float:
        """Patch resonant length L (along y) [m]."""
        return patch_design(self.f0, self.eps_r, self.h_sub)[1]

    @property
    def eps_reff(self) -> float:
        """Effective permittivity of the microstrip line [-]."""
        return patch_design(self.f0, self.eps_r, self.h_sub)[2]

    @property
    def sub_w(self) -> float:
        """Substrate / ground-plane extent along x [m]."""
        return self.patch_w + 2.0 * self.ground_margin

    @property
    def sub_l(self) -> float:
        """Substrate / ground-plane extent along y [m]."""
        return self.patch_l + 2.0 * self.ground_margin

    def feed_offset_y(self) -> float:
        """Inset of the probe feed from the near radiating edge [m].

        gradenna feeds one cell in from the patch edge (pj = j0 + 1; the gap
        feed sits at the first free Ez layer). At the benchmark cell size
        (1.2 mm) that is ~1.2 mm; we expose it as a length so the openEMS
        model can place its probe at the same physical inset. openEMS will
        round it to its own mesh.
        """
        return 1.2e-3

    @property
    def conductivity_sub(self) -> float:
        """FR-4 bulk conductivity from the loss tangent at f0 [S/m].

        sigma = 2 pi f0 eps0 eps_r tan_d. NOTE: this is the *physical* FR-4
        conductivity. gradenna additionally scales eps_r (and hence sigma) by
        the pin-layer factor (n_free/n_gap) -- that is a discretisation
        correction internal to its thin-sheet metal model and does NOT change
        the physical antenna described here. openEMS, which resolves the
        substrate thickness with several cells and real metal sheets, uses the
        unscaled values below.
        """
        return 2.0 * math.pi * self.f0 * EPS0 * self.eps_r * self.tan_d


@dataclass(frozen=True)
class GradennaMesh:
    """FDTD discretisation for the gradenna side (mirrors test_patch_antenna).

    These are intentionally identical to the constants at the top of
    ``tests/test_patch_antenna.py`` so the cross-check reproduces the same
    grid that the analytic-resonance test already validates.
    """

    dxy: float = 1.2e-3          # in-plane cell (~ lambda0/100 at 2.45 GHz)
    n_gap: int = 3               # ground-to-patch cell layers (= h_sub/dz)
    m_gnd: int = 8               # ground/substrate margin beyond patch [cells]
    m_air: int = 5               # air gap between ground edge and CPML [cells]
    n_pml: int = 8               # CPML thickness [cells]
    n_steps: int = 6400          # ~9.5 ns: pulse + lossy ringdown
    sig_metal: float = 1.0e7     # thin-sheet PEC surrogate [S/m]

    @property
    def dz(self) -> float:
        """Vertical cell size [m] (substrate height split into n_gap cells)."""
        return H_SUB / self.n_gap


# A ready-made default instance for callers that just want "the benchmark".
PATCH = PatchGeometry()
GRADENNA_MESH = GradennaMesh()
