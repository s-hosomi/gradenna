# PCB Fabrication and NanoVNA Measurement Runbook

Campaign: 2.45 GHz FR-4 Patch Antenna — Physical Verification

## Overview

This campaign validates the gradenna FDTD simulation against a fabricated
prototype. The target antenna is the **openEMS benchmark patch**
(`benchmarks/openems_refs/geometry.py`), which is the same shape used for
all simulation cross-checks. Measuring this exact geometry closes the loop:
sim vs. openEMS vs. physical prototype.

**Feed type: probe (coaxial).** The gradenna simulation and the openEMS
reference model both use a probe feed placed 1.2 mm inset from the near
radiating edge, centred along W. The PCB reflects this exactly: no feed pad
stub, no edge-launch connector. An SMA flange/panel-mount connector is
soldered to the ground side with its centre pin passed through the plated
drill hole and soldered on the patch top face.

**Impedance note:** This probe position was chosen to match the simulation
model, not to optimise 50 Ω return loss. The antenna will **not** be perfectly
50 Ω matched. The pass/fail criterion is agreement between simulation and
measurement (resonant frequency within ±2 %, |S11| shape similarity), **not**
a specific |S11| depth at resonance. This mirrors the criterion used in
`tests/test_patch_antenna.py`, which checks resonant frequency against the
analytic Balanis design value.

---

## Antenna Dimensions (Balanis TL design)

All dimensions derived from `benchmarks/openems_refs/geometry.py` (single
source of truth; do not change independently).

| Parameter | Value |
|-----------|-------|
| Design frequency | 2.45 GHz |
| Substrate | FR-4, eps_r = 4.3, h = 1.6 mm, tan_delta = 0.02 |
| Patch width W | 37.5839 mm |
| Patch length L | 29.1383 mm |
| eps_reff | 3.9924 |
| Board size | 47.5839 x 39.1383 mm |
| Margin (each side) | 5.0 mm |

## Feed Point Coordinates

The probe feed is placed at the W-centre, 1.2 mm inset from the near
radiating edge (y = MARGIN_MM = 5.0 mm side). Calculation:

```
feed_x = MARGIN_MM + W/2  = 5.0 + 37.5839/2 = 23.7919 mm
feed_y = MARGIN_MM + 1.2  = 5.0 + 1.2       =  6.2000 mm
```

The 1.2 mm inset equals `PatchGeometry.feed_offset_y()` in geometry.py
(one FDTD cell at the benchmark grid pitch of 1.2 mm/cell).

| Feature | x (mm) | y (mm) |
|---------|---------|--------|
| Patch corner (near, left) | 5.0000 | 5.0000 |
| Patch corner (far, right) | 42.5839 | 34.1383 |
| Feed / drill / anti-pad centre | **23.7919** | **6.2000** |

---

## Gerber / Drill Files

All files are in `fab_campaign/gerber/`. Regenerate with:

```sh
# From the repo root
.venv/bin/python fab_campaign/generate_fab.py
```

| File | Layer / Function | Description |
|------|-----------------|-------------|
| `patch_top_copper.gbr` | Copper, L1, Top | Patch rectangle only — no stub |
| `patch_bottom_copper.gbr` | Copper, L2, Bot | Full ground plane with 4 mm dia anti-pad void |
| `patch_edge_cuts.gbr` | Profile, NP | Board outline 47.58 x 39.14 mm |
| `patch_top_soldermask.gbr` | Soldermask, Top | Mask opening matching patch rectangle |
| `patch_bot_soldermask.gbr` | Soldermask, Bot | 12 mm dia mask opening at anti-pad centre |
| `patch.drl` | Excellon drill | 1.6 mm NPTH at feed point (23.792, 6.200) mm |
| `patch_top_copper.svg` | — | SVG preview of top copper |
| `patch_bottom_copper.svg` | — | SVG preview of bottom copper |

### Anti-pad Detail

The bottom copper has a circular void (anti-pad) centred exactly on the feed
point. Dimensions:
- Anti-pad diameter: **4.0 mm** (clearance from centre pin to ground copper)
- Bottom soldermask opening: **12.0 mm diameter**, exposing the ground
  copper around the anti-pad so the SMA flange body can be soldered to it
  (a smaller opening would leave too thin an annulus of exposed copper)

---

## Part 1: JLCPCB Order Instructions

### Building the ZIP

Zip all Gerber and drill files together:

```sh
cd fab_campaign/gerber
zip ../patch_fab_jlcpcb.zip \
    patch_top_copper.gbr \
    patch_bottom_copper.gbr \
    patch_edge_cuts.gbr \
    patch_top_soldermask.gbr \
    patch_bot_soldermask.gbr \
    patch.drl
```

Upload `fab_campaign/patch_fab_jlcpcb.zip` to JLCPCB.

### JLCPCB Order Settings

| Setting | Value |
|---------|-------|
| PCB layers | 2 |
| PCB thickness | **1.6 mm** |
| Copper weight | **1 oz (35 µm)** |
| Surface finish | HASL (lead-free) or ENIG |
| Material | FR-4 (default, TG 135) |
| Board size | 47.58 x 39.14 mm (auto-detected from outline) |
| Min trace/clearance | 5/5 mil (JLCPCB standard — DRC passed) |

Do not enable panelisation or V-cut; the board is a single outline.

### Upload Procedure

1. Go to jlcpcb.com -> Instant Quote.
2. Click "Add Gerber File" and upload `patch_fab_jlcpcb.zip`.
3. Confirm the board size is detected as ~47.6 x 39.1 mm.
4. Set the parameters from the table above; all other settings can remain
   default.
5. Quantity: 5 (minimum JLCPCB order; gives spare boards).
6. Add to cart and complete the order.

### SMA Connector and Assembly

**Connector type:** SMA flange/panel-mount (not edge-launch). Choose a
vertical through-hole style with a 4-bolt flange such as:
- Amphenol 132289 or equivalent panel-mount SMA, rated for 1.6 mm PCB.
- The flange bolt holes are **not drilled** in this PCB (to keep the design
  simple for a prototype). Solder the flange tabs to the bottom (ground)
  copper plane with a small amount of solder fillet. If mechanical rigidity is
  needed, add small blobs of epoxy under the flange tabs.

**Assembly steps:**
1. Insert the SMA centre pin from the ground (bottom) side through the
   1.6 mm drill hole.
2. Solder the pin on the **patch top face** with a minimal solder dot
   (do not bridge to the patch edge or create a large pad — the simulation
   has no such pad).
3. Solder the SMA flange/shell to the **bottom copper ground plane**.
4. No other components are needed.

---

## Part 2: NanoVNA Measurement Procedure

### Equipment

- NanoVNA (V2 or later) with calibration kit (SHORT, OPEN, LOAD).
- USB cable for pynanovna, or NanoVNA-Saver GUI as an alternative.
- SMA cable (< 30 cm to minimise loss; re-calibrate if cable is changed).

### Setup and Calibration

1. **Warm-up**: Power on the NanoVNA for at least 5 minutes.
2. **Sweep range**: 2.0 GHz to 3.0 GHz, 201 points.
   (This covers the expected resonance and the first sidelobe.)
3. **SOLT calibration** (one-port: S, O, L):
   a. Attach the SHORT standard to Port 0 of the NanoVNA.
      On the device, select Calibrate -> Short; wait for the sweep.
   b. Attach the OPEN standard; select Calibrate -> Open.
   c. Attach the 50 ohm LOAD standard; select Calibrate -> Load.
   d. Apply/Save calibration. Do NOT detach or flex the cable after this.
4. **Reference plane**: calibrate at the connector end of the cable that
   will attach to the antenna SMA. If you add an adapter, calibrate with
   the adapter in place.

### Capture with `scripts/nanovna_capture.py`

Install pynanovna first (it is GPLv3, not a gradenna dependency):

```sh
.venv/bin/pip install pynanovna
```

Then run:

```sh
.venv/bin/python scripts/nanovna_capture.py \
    --start 2.0e9 --stop 3.0e9 --points 201 \
    --out fab_campaign/measured/patch_s11.s1p
```

If you have a saved calibration file, add `--calibration <path>`.

**Alternative (no pynanovna):** Use NanoVNA-Saver GUI, sweep 2.0–3.0 GHz,
and export the result as a Touchstone `.s1p` file. Place it at
`fab_campaign/measured/patch_s11.s1p`.

### Measurement Checklist

- [ ] NanoVNA warmed up >= 5 minutes
- [ ] Calibration performed at the cable end (not at NanoVNA port)
- [ ] Antenna held in free space (> 10 cm from any metal surface or hand)
- [ ] `patch_s11.s1p` saved with 201 points, 2.0–3.0 GHz

---

## Part 3: Sim-vs-Measured Comparison with `gradenna.measure`

### Prerequisite: Simulation S11

Run the benchmark simulation to produce the reference S11 curve. This is
the same run that tests/test_patch_antenna.py validates:

```python
# Minimal snippet -- see tests/test_patch_antenna.py for the full setup
import numpy as np
from gradenna.measure import save_touchstone

# f_sweep and s11_sim come from the gradenna FDTD run (sparams module).
# Run the simulation and capture them, then:
save_touchstone(
    "fab_campaign/simulated/patch_s11_sim.s1p",
    f_sweep, s11_sim
)
```

Or reuse the curve already computed during the openEMS cross-check
(benchmarks/openems_refs/generate_patch_refs.py), which runs the same
gradenna model and can write its S11 to a file.

### Running the Comparison

```python
from gradenna.measure import load_touchstone, compare_s11, plot_s11_comparison

sim  = load_touchstone("fab_campaign/simulated/patch_s11_sim.s1p")
meas = load_touchstone("fab_campaign/measured/patch_s11.s1p")

result = compare_s11(
    sim.f, sim.s[:, 0, 0],
    meas,
    band=(2.0e9, 3.0e9)
)

print(f"f_res_sim  = {result.f_res_sim / 1e9:.4f} GHz")
print(f"f_res_meas = {result.f_res_meas / 1e9:.4f} GHz")
print(f"shift      = {result.f_res_shift_pct:+.2f} %")
print(f"S11 RMS diff = {result.rms_diff_db:.2f} dB")

plot_s11_comparison(
    sim.f, sim.s[:, 0, 0],
    meas,
    path="fab_campaign/results/s11_comparison.png"
)
```

### Pass/Fail Criteria

The pass criterion is **sim-vs-measured agreement**, not an absolute
return-loss target. The probe feed is not impedance-matched to 50 Ω by
design; its position was chosen to replicate the simulation model exactly.

| Metric | Pass threshold | Notes |
|--------|---------------|-------|
| Resonance shift `f_res_shift_pct` | <= ±2 % | ~49 MHz at 2.45 GHz |
| |S11| RMS difference (sim vs meas) | <= 2 dB | Over 2.0–3.0 GHz band |

A resonance shift larger than ±2 % most likely indicates FR-4 permittivity
variation (typical batch eps_r = 4.2–4.5 instead of 4.3) or SMA connector
de-embedding error. If the shift exceeds 5 %, re-check calibration and
connector placement before concluding a modelling issue.

---

## Part 4: DRC Results

DRC performed by `gradenna.fab.check_min_width` / `check_min_gap`
against JLCPCB 5/5 mil (0.127 mm) rules at time of Gerber generation:

| Layer | Width violations | Gap violations |
|-------|-----------------|----------------|
| Top copper (patch rect) | 0 | 0 |
| Bottom copper (ground + anti-pad) | 0 | 0 |

The anti-pad circle (4 mm dia) and soldermask circles are single connected
regions with no copper-to-copper gap; the width everywhere exceeds the
5 mil minimum by a large margin.

---

## Part 5: Open Items for the User (Physical Tasks)

These items cannot be automated and require user action:

- [ ] **Build the ZIP** (see instructions above) and upload to JLCPCB.
- [ ] **Order SMA flange/panel-mount connector** (50 ohm, 1.6 mm PCB,
      vertical through-hole, e.g. Amphenol 132289 or equivalent).
- [ ] **Solder SMA connector** when boards arrive (pin through bottom,
      solder on patch top face; flange shell soldered to ground plane).
- [ ] **Install pynanovna** (`pip install pynanovna`) and verify USB
      connection to the NanoVNA before the measurement session.
- [ ] **Perform SOLT calibration** at the cable end, then run
      `scripts/nanovna_capture.py` or export from NanoVNA-Saver.
- [ ] **Save measurement file** to `fab_campaign/measured/patch_s11.s1p`.
- [ ] **Run the simulation** (`tests/test_patch_antenna.py` or the openEMS
      cross-check script) to produce `fab_campaign/simulated/patch_s11_sim.s1p`.
- [ ] **Run `compare_s11`** and check against the pass/fail criteria above.
- [ ] **Commit results** (measured .s1p, comparison PNG, summary) to
      `benchmarks/openems_refs/` or a new `fab_campaign/results/` directory
      as appropriate.

### Known Caveats

- The openEMS reference CSVs (`benchmarks/openems_refs/s11.csv` etc.) are
  not yet committed (see `benchmarks/openems_refs/README.md`). Once they
  exist, the pass criterion can be tightened to match the openEMS tolerance.
- The gradenna model uses a simplified thin-sheet metal (pin-layer) model
  for the substrate. The physical FR-4 permittivity batch variation
  (eps_r = 4.2–4.5) is the dominant uncertainty and is expected to account
  for most of the resonance shift.
- `gradenna.measure` requires scikit-rf (`pip install scikit-rf` or the
  `[measure]` extra). It is already in the `dev` dependency group.
- The SMA flange bolt holes are not drilled. Use solder or epoxy to
  mechanically secure the connector flange to the ground plane.
