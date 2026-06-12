// Perceptual colormap LUTs and canvas helpers.
//
// All maps are built as 256-entry RGB lookup tables from anchor stops of the
// reference colormaps (matplotlib viridis / RdBu) or hand-tuned stops, so the
// per-pixel hot loops are a single table lookup instead of branchy math.

export type RGB = [number, number, number];

function buildLUT(stops: [number, RGB][], n = 256): Uint8Array {
  const lut = new Uint8Array(n * 3);
  for (let k = 0; k < n; k++) {
    const t = k / (n - 1);
    let i = 0;
    while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    const s = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 0;
    // smoothstep easing between anchors avoids visible Mach bands
    const e = s * s * (3 - 2 * s);
    lut[k * 3] = Math.round(c0[0] + e * (c1[0] - c0[0]));
    lut[k * 3 + 1] = Math.round(c0[1] + e * (c1[1] - c0[1]));
    lut[k * 3 + 2] = Math.round(c0[2] + e * (c1[2] - c0[2]));
  }
  return lut;
}

/** matplotlib viridis (9 anchors, smooth-lerped). */
export const VIRIDIS = buildLUT([
  [0.0, [68, 1, 84]],
  [0.125, [71, 44, 122]],
  [0.25, [59, 81, 139]],
  [0.375, [44, 113, 142]],
  [0.5, [33, 144, 141]],
  [0.625, [39, 173, 129]],
  [0.75, [92, 200, 99]],
  [0.875, [170, 220, 50]],
  [1.0, [253, 231, 37]],
]);

/** Density map: dark substrate -> bronze -> bright copper highlight. */
export const COPPER = buildLUT([
  [0.0, [11, 14, 22]],
  [0.3, [30, 39, 60]],
  [0.55, [92, 64, 47]],
  [0.75, [190, 108, 44]],
  [0.9, [233, 148, 60]],
  [1.0, [255, 216, 164]],
]);

/** matplotlib RdBu reversed: deep blue (-1) -> near-white (0) -> deep red (+1). */
export const RDBU = buildLUT([
  [0.0, [12, 51, 96]],
  [0.18, [50, 110, 168]],
  [0.34, [126, 173, 210]],
  [0.46, [200, 218, 230]],
  [0.5, [234, 236, 240]],
  [0.54, [233, 213, 201]],
  [0.66, [219, 153, 125]],
  [0.82, [186, 81, 64]],
  [1.0, [113, 14, 33]],
]);

/**
 * Dark-centred diverging map for fields on a dark UI: glowing blue for
 * negative, near-background at zero, glowing copper for positive.
 */
export const DARK_DIV = buildLUT([
  [0.0, [96, 168, 255]],
  [0.22, [38, 78, 150]],
  [0.42, [15, 20, 32]],
  [0.5, [11, 14, 20]],
  [0.58, [34, 22, 14]],
  [0.78, [168, 88, 34]],
  [1.0, [255, 178, 92]],
]);

/** Sample a LUT at t in [0,1]. */
export function lutColor(lut: Uint8Array, t: number): RGB {
  const v = Math.max(0, Math.min(1, t));
  const k = Math.round(v * 255) * 3;
  return [lut[k], lut[k + 1], lut[k + 2]];
}

export function lutCSS(lut: Uint8Array, t: number): string {
  const [r, g, b] = lutColor(lut, t);
  return `rgb(${r},${g},${b})`;
}

/** Build a 256x1 RGBA byte strip of a LUT (for shader colormap textures). */
export function lutToRGBA(lut: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(new ArrayBuffer(256 * 4));
  for (let k = 0; k < 256; k++) {
    out[k * 4] = lut[k * 3];
    out[k * 4 + 1] = lut[k * 3 + 1];
    out[k * 4 + 2] = lut[k * 3 + 2];
    out[k * 4 + 3] = 255;
  }
  return out;
}

/**
 * Fill an RGBA pixel buffer from signed values using a diverging LUT with a
 * tanh soft-clip: t = 0.5 + 0.5*tanh(v / (softness*scale)). Keeps the
 * mid-amplitude wave structure vivid instead of letting the source peak
 * dominate the normalization.
 */
export function fillDivergingRGBA(
  out: Uint8Array,
  values: ArrayLike<number>,
  n: number,
  scale: number,
  lut: Uint8Array = RDBU,
  softness = 0.35
): void {
  const inv = 1 / Math.max(scale * softness, 1e-12);
  for (let k = 0; k < n; k++) {
    const x = (values[k] as number) * inv;
    // cheap tanh
    const e2 = Math.exp(2 * Math.max(-20, Math.min(20, x)));
    const t = 0.5 + 0.5 * ((e2 - 1) / (e2 + 1));
    const idx = Math.round(t * 255) * 3;
    const p = k * 4;
    out[p] = lut[idx];
    out[p + 1] = lut[idx + 1];
    out[p + 2] = lut[idx + 2];
    out[p + 3] = 255;
  }
}

/** Draw a vertical colorbar with tick labels into a canvas (HiDPI aware). */
export function drawColorbar(
  canvas: HTMLCanvasElement,
  lut: Uint8Array,
  ticks: { t: number; label: string }[],
  title = ""
): void {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 56;
  const cssH = canvas.clientHeight || 180;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);

  const padTop = title ? 18 : 6;
  const padBot = 6;
  const barW = 10;
  const barX = 2;
  const barH = cssH - padTop - padBot;

  for (let y = 0; y < barH; y++) {
    const t = 1 - y / (barH - 1);
    ctx.fillStyle = lutCSS(lut, t);
    ctx.fillRect(barX, padTop + y, barW, 1.5);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1;
  ctx.strokeRect(barX + 0.5, padTop + 0.5, barW - 1, barH - 1);

  ctx.fillStyle = "#7a8499";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "middle";
  for (const { t, label } of ticks) {
    const y = padTop + (1 - t) * (barH - 1);
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.fillRect(barX + barW, y, 3, 1);
    ctx.fillStyle = "#9aa3b5";
    ctx.fillText(label, barX + barW + 6, y);
  }
  if (title) {
    ctx.fillStyle = "#7a8499";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(title, 0, 12);
  }
}
