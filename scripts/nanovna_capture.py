#!/usr/bin/env python
"""Capture S11 from a NanoVNA and save it as a Touchstone .s1p file.

Uses pynanovna (https://github.com/PICC-Group/pynanovna) when available.
pynanovna is GPLv3-licensed, so it is deliberately NOT a dependency of
gradenna (MIT); install it manually into your environment if you want
hardware capture:

    pip install pynanovna

Without pynanovna this script prints usage instructions and exits.

Example:

    python scripts/nanovna_capture.py \\
        --start 2.0e9 --stop 3.0e9 --points 201 --out antenna.s1p

SOLT calibration procedure (one-port: S, O, L are what matters for S11):
  1. Let the NanoVNA warm up for a few minutes (thermal drift).
  2. On the device or via pynanovna's calibration helper
     (see pynanovna's examples/example_calibration.py), sweep the band of
     interest, then attach the calibration standards to port 0 in turn:
       - SHORT standard  -> record "short"
       - OPEN  standard  -> record "open"
       - 50 ohm LOAD     -> record "load"
     (ISOLATION and THROUGH are only needed for S21 / two-port work.)
  3. Save the calibration to a file so it can be reloaded with
     --calibration on subsequent runs.
  4. Calibrate at the same reference plane as the antenna connector
     (i.e. include any adapter/cable you will measure through).
  5. Re-calibrate whenever the sweep range, cable, or temperature changes.
"""

from __future__ import annotations

import argparse
import sys

try:  # pynanovna is optional (GPLv3 -- never add it to gradenna's deps).
    import pynanovna
except ImportError:
    pynanovna = None

_NO_PYNANOVNA_MSG = """\
pynanovna is not installed, so hardware capture is unavailable.

pynanovna is GPLv3 and therefore not a gradenna dependency. To use this
script, install it manually:

    pip install pynanovna

Then connect a NanoVNA via USB and run, for example:

    python scripts/nanovna_capture.py --start 2.0e9 --stop 3.0e9 \\
        --points 201 --out antenna.s1p

Alternative without pynanovna: export a Touchstone .s1p from the
NanoVNA-Saver GUI and load it with gradenna.measure.load_touchstone().
"""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture S11 from a NanoVNA and save a Touchstone .s1p file.",
    )
    parser.add_argument("--start", type=float, required=True, help="sweep start frequency [Hz], e.g. 2.0e9")
    parser.add_argument("--stop", type=float, required=True, help="sweep stop frequency [Hz], e.g. 3.0e9")
    parser.add_argument("--points", type=int, default=201, help="number of sweep points (default: 201)")
    parser.add_argument("--out", type=str, required=True, help="output Touchstone file (.s1p)")
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="optional pynanovna SOLT calibration file to load before sweeping",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if pynanovna is None:
        print(_NO_PYNANOVNA_MSG, file=sys.stderr)
        return 1

    if args.start <= 0 or args.stop <= args.start:
        print("error: require 0 < --start < --stop", file=sys.stderr)
        return 2

    # Imported lazily so the usage message above works even in a bare env.
    from gradenna.measure import save_touchstone

    vna = pynanovna.VNA()

    if args.calibration is not None:
        # Calibration data previously recorded with pynanovna's SOLT helper
        # (see module docstring for the recording procedure).
        vna.load_calibration(args.calibration)

    try:
        vna.set_sweep(args.start, args.stop, args.points)
    except TypeError:
        # Older pynanovna versions take only (start, stop); the point count
        # is then whatever the device firmware is configured for.
        vna.set_sweep(args.start, args.stop)

    s11, _s21, freqs = vna.sweep()
    vna.kill()

    save_touchstone(args.out, freqs, s11)
    print(f"wrote {len(freqs)} points to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
