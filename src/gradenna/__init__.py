"""gradenna — differentiable FDTD antenna inverse design in JAX.

Submodules with optional heavy dependencies are not imported here:
`gradenna.fab` (scikit-image / shapely / gerber-writer, extra "fab") and
`gradenna.measure` (scikit-rf, extra "measure").
"""

from gradenna.constants import C0, EPS0, ETA0, MU0
from gradenna.cpml import CPMLSpec, alpha_max_for_fmin
from gradenna.designs import patch_design
from gradenna.estimate import fdtd3d_memory_estimate, fits_gpu, gpu_fit_report
from gradenna.fdtd2d import Port, SimResult, field_energy, simulate_tm
from gradenna.freq_adjoint import (
    FreqPhasors,
    exact_design_gradient,
    freq_adjoint_gradient,
    simulate_tm_freq,
)
from gradenna.fdtd3d import (
    SimResult3D,
    field_energy_3d,
    port_impedance,
    simulate_3d,
    time_series_dft,
)
from gradenna.grid import Grid2D, Grid3D
from gradenna.materials import sheet_conductivity, sigma_from_density
from gradenna.monitors import log_radiated_fraction, poynting_flux_box_2d
from gradenna.ntff import (
    directivity_2d,
    directivity_3d,
    gain,
    ntff_2d,
    ntff_3d,
    radiated_power_2d,
    radiated_power_3d,
)
from gradenna.sources import gaussian, gaussian_derivative, modulated_gaussian
from gradenna.sparams import (
    BandPulse,
    gaussian_pulse_for_band,
    half_step_dft,
    incident_voltage,
    port_dft,
    s11_power_wave,
)
from gradenna.topopt import (
    DesignTransform,
    beta_schedule,
    conic_filter,
    connected_to_seed,
    gray_indicator,
    minimum_feature_size,
    optimize,
    tanh_projection,
)

__all__ = [
    "C0",
    "EPS0",
    "ETA0",
    "MU0",
    "BandPulse",
    "CPMLSpec",
    "DesignTransform",
    "FreqPhasors",
    "Grid2D",
    "Grid3D",
    "Port",
    "SimResult",
    "SimResult3D",
    "alpha_max_for_fmin",
    "beta_schedule",
    "conic_filter",
    "connected_to_seed",
    "directivity_2d",
    "directivity_3d",
    "exact_design_gradient",
    "fdtd3d_memory_estimate",
    "field_energy",
    "field_energy_3d",
    "fits_gpu",
    "freq_adjoint_gradient",
    "gain",
    "gpu_fit_report",
    "gaussian",
    "gaussian_derivative",
    "gaussian_pulse_for_band",
    "gray_indicator",
    "half_step_dft",
    "incident_voltage",
    "log_radiated_fraction",
    "minimum_feature_size",
    "modulated_gaussian",
    "ntff_2d",
    "ntff_3d",
    "optimize",
    "patch_design",
    "port_dft",
    "port_impedance",
    "poynting_flux_box_2d",
    "radiated_power_2d",
    "radiated_power_3d",
    "s11_power_wave",
    "sheet_conductivity",
    "sigma_from_density",
    "simulate_3d",
    "simulate_tm",
    "simulate_tm_freq",
    "tanh_projection",
    "time_series_dft",
]
