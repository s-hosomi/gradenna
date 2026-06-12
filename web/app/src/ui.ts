// Shared design tokens and small DOM helpers used by every view.

export const T = {
  bg: "#0b0e14",
  panel: "#11151f",
  panelEdge: "#1d2433",
  text: "#c9d1e0",
  dim: "#7a8499",
  faint: "#454e61",
  accent: "#5aa2ff",
  accentSoft: "rgba(90,162,255,0.14)",
  copper: "#e8943c",
  copperSoft: "rgba(232,148,60,0.16)",
  mono: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
  sans: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, sans-serif",
} as const;

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  css = "",
  text = ""
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (css) e.style.cssText = css;
  if (text) e.textContent = text;
  return e;
}

/** Rounded panel with the standard surface treatment. */
export function panel(extra = ""): HTMLDivElement {
  return el(
    "div",
    `background:${T.panel};border:1px solid ${T.panelEdge};border-radius:10px;` +
      `box-shadow:0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.35);` +
      extra
  );
}

export function sectionTitle(text: string, sub = ""): HTMLDivElement {
  const wrap = el("div", "display:flex;align-items:baseline;gap:10px;");
  wrap.appendChild(
    el(
      "h2",
      `margin:0;font-size:0.95rem;font-weight:600;letter-spacing:0.01em;color:${T.text};font-family:${T.sans};`,
      text
    )
  );
  if (sub)
    wrap.appendChild(
      el("span", `font-size:0.72rem;color:${T.dim};font-family:${T.mono};`, sub)
    );
  return wrap;
}

export interface BtnOpts {
  primary?: boolean;
  active?: boolean;
}

export function button(label: string, opts: BtnOpts = {}): HTMLButtonElement {
  const b = el("button") as HTMLButtonElement;
  b.textContent = label;
  styleButton(b, opts);
  b.addEventListener("mouseenter", () => (b.style.filter = "brightness(1.18)"));
  b.addEventListener("mouseleave", () => (b.style.filter = ""));
  return b;
}

export function styleButton(b: HTMLButtonElement, opts: BtnOpts = {}): void {
  const { primary = false, active = false } = opts;
  const bg = active ? T.accentSoft : primary ? "rgba(90,162,255,0.12)" : "rgba(255,255,255,0.04)";
  const edge = active || primary ? "rgba(90,162,255,0.55)" : T.panelEdge;
  const fg = active ? T.accent : T.text;
  b.style.cssText =
    `background:${bg};border:1px solid ${edge};color:${fg};` +
    `padding:5px 12px;border-radius:7px;cursor:pointer;font-size:0.78rem;` +
    `font-family:${T.sans};font-weight:500;transition:filter .12s, border-color .12s, background .12s;` +
    `white-space:nowrap;`;
}

/** Small key/value readout row (monospace value). */
export function statRow(key: string): { row: HTMLDivElement; value: HTMLSpanElement } {
  const row = el(
    "div",
    `display:flex;justify-content:space-between;gap:12px;font-size:0.74rem;` +
      `color:${T.dim};font-family:${T.sans};padding:2px 0;`
  );
  row.appendChild(el("span", "", key));
  const value = el("span", `color:${T.text};font-family:${T.mono};`) as HTMLSpanElement;
  row.appendChild(value);
  return { row, value };
}

/** Centered overlay message (data-missing fallbacks). */
export function overlayMessage(host: HTMLElement, html: string): HTMLDivElement {
  host.style.position = "relative";
  const o = el(
    "div",
    `position:absolute;inset:0;display:flex;align-items:center;justify-content:center;` +
      `text-align:center;color:${T.dim};font-size:0.85rem;font-family:${T.sans};line-height:1.9;` +
      `background:rgba(11,14,20,0.6);backdrop-filter:blur(2px);z-index:5;`
  );
  const inner = el("div");
  inner.innerHTML = html;
  o.appendChild(inner);
  host.appendChild(o);
  return o;
}

/** Prepare a canvas for HiDPI drawing; returns ctx scaled to CSS pixels. */
export function hidpiCtx(canvas: HTMLCanvasElement): {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
} | null {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w < 1 || h < 1) return null;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

export function formatSI(v: number, unit: string, digits = 2): string {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(digits)} G${unit}`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(digits)} M${unit}`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(digits)} k${unit}`;
  if (a >= 1) return `${v.toFixed(digits)} ${unit}`;
  if (a >= 1e-3) return `${(v * 1e3).toFixed(digits)} m${unit}`;
  if (a >= 1e-6) return `${(v * 1e6).toFixed(digits)} µ${unit}`;
  if (a >= 1e-9) return `${(v * 1e9).toFixed(digits)} n${unit}`;
  return `${v.toExponential(digits)} ${unit}`;
}
