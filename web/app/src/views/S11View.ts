import { S11Data } from "../types";
import { T, el, panel, sectionTitle, statRow, overlayMessage, hidpiCtx } from "../ui";

const C_GRAD = "#5aa2ff";
const C_OEMS = "#ffb454";

export class S11View {
  private container: HTMLElement;
  private canvas: HTMLCanvasElement | null = null;
  private data: S11Data | null = null;
  private ro: ResizeObserver | null = null;
  private hoverX: number | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  async init(): Promise<void> {
    const c = this.container;
    c.innerHTML = "";
    c.style.cssText =
      "display:flex;flex-direction:column;height:100%;gap:12px;padding:18px 20px;box-sizing:border-box;";
    c.appendChild(
      sectionTitle("S11 return loss", "gradenna vs. an independent solver (openEMS)")
    );

    const row = el("div", "display:flex;gap:14px;flex:1;min-height:0;");
    c.appendChild(row);

    const plotPanel = panel("flex:1;min-width:0;overflow:hidden;position:relative;padding:6px;");
    row.appendChild(plotPanel);
    this.canvas = el("canvas", "width:100%;height:100%;display:block;cursor:crosshair;") as HTMLCanvasElement;
    plotPanel.appendChild(this.canvas);

    const side = el("div", "width:230px;flex-shrink:0;display:flex;flex-direction:column;gap:10px;");
    row.appendChild(side);
    const hud = panel("padding:12px;display:flex;flex-direction:column;gap:2px;");
    side.appendChild(hud);
    const sLabel = statRow("antenna");
    const sDipG = statRow("dip · gradenna");
    const sDipO = statRow("dip · openEMS");
    const sAgree = statRow("agreement");
    for (const s of [sLabel, sDipG, sDipO, sAgree]) hud.appendChild(s.row);

    const hint = el(
      "div",
      `color:${T.faint};font-size:0.72rem;line-height:1.7;font-family:${T.sans};padding:0 2px;`
    );
    hint.innerHTML =
      "The reference CSVs are committed<br>to the repo and compared in CI:<br>resonance ±2%, |S11| RMS ≤ 2 dB,<br>pattern correlation ≥ 0.99.";
    side.appendChild(hint);

    try {
      const resp = await fetch("./data/s11.json");
      if (!resp.ok) throw new Error(String(resp.status));
      this.data = (await resp.json()) as S11Data;
    } catch {
      overlayMessage(
        plotPanel,
        `No data yet —<br><code style="color:${T.copper}">python scripts/export_viz.py</code>`
      );
      return;
    }

    const d = this.data;
    const dip = (ys: number[]) => {
      let k = 0;
      ys.forEach((v, i) => { if (v < ys[k]) k = i; });
      return { f: d.freq_hz[k], db: ys[k] };
    };
    const dg = dip(d.s11_db_gradenna);
    sLabel.value.textContent = d.label;
    sDipG.value.textContent = `${(dg.f / 1e9).toFixed(3)} GHz`;
    if (d.s11_db_openems) {
      const doo = dip(d.s11_db_openems);
      sDipO.value.textContent = `${(doo.f / 1e9).toFixed(3)} GHz`;
      sAgree.value.textContent = `${((Math.abs(dg.f - doo.f) / doo.f) * 100).toFixed(2)} %`;
    } else {
      sDipO.value.textContent = "—";
      sAgree.value.textContent = "—";
    }

    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.canvas);
    this.canvas.addEventListener("pointermove", (e: PointerEvent) => {
      const rect = this.canvas!.getBoundingClientRect();
      this.hoverX = e.clientX - rect.left;
      this.draw();
    });
    this.canvas.addEventListener("pointerleave", () => {
      this.hoverX = null;
      this.draw();
    });
    this.draw();
  }

  private draw(): void {
    if (!this.canvas || !this.data) return;
    const res = hidpiCtx(this.canvas);
    if (!res) return;
    const { ctx, w, h } = res;
    const d = this.data;

    const pad = { top: 26, right: 18, bottom: 42, left: 52 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;

    const fMin = d.freq_hz[0];
    const fMax = d.freq_hz[d.freq_hz.length - 1];
    const yMin = -42;
    const yMax = 2;
    const toX = (f: number) => pad.left + ((f - fMin) / (fMax - fMin)) * cw;
    const toY = (db: number) => pad.top + ((yMax - db) / (yMax - yMin)) * ch;

    ctx.clearRect(0, 0, w, h);

    // matched region (below -10 dB)
    ctx.fillStyle = "rgba(90,200,120,0.05)";
    ctx.fillRect(pad.left, toY(-10), cw, toY(yMin) - toY(-10));

    // grid
    ctx.font = `10px ${T.mono}`;
    ctx.fillStyle = T.faint;
    ctx.strokeStyle = "rgba(255,255,255,0.055)";
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let db = 0; db >= -40; db -= 10) {
      const y = toY(db);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + cw, y);
      ctx.stroke();
      ctx.fillText(`${db}`, pad.left - 8, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const ghzStep = 0.25e9;
    for (let f = Math.ceil(fMin / ghzStep) * ghzStep; f <= fMax; f += ghzStep) {
      const x = toX(f);
      ctx.strokeStyle = "rgba(255,255,255,0.04)";
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + ch);
      ctx.stroke();
      ctx.fillStyle = T.faint;
      ctx.fillText((f / 1e9).toFixed(2), x, pad.top + ch + 8);
    }

    // -10 dB reference
    const yRef = toY(-10);
    ctx.strokeStyle = "rgba(90,200,120,0.4)";
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(pad.left, yRef);
    ctx.lineTo(pad.left + cw, yRef);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(90,200,120,0.6)";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText("−10 dB", pad.left + 6, yRef - 3);

    // axis titles
    ctx.fillStyle = T.dim;
    ctx.font = `11px ${T.sans}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText("frequency (GHz)", pad.left + cw / 2, h - 8);
    ctx.save();
    ctx.translate(14, pad.top + ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("|S11|  (dB)", 0, 0);
    ctx.restore();

    // curves (glow underlay + crisp line)
    const drawCurve = (ys: number[], color: string) => {
      for (const [width, alpha] of [[5, 0.12], [1.8, 1]] as [number, number][]) {
        ctx.beginPath();
        ys.forEach((v, i) => {
          const x = toX(d.freq_hz[i]);
          const y = toY(Math.max(yMin, v));
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = color;
        ctx.globalAlpha = alpha;
        ctx.lineWidth = width;
        ctx.lineJoin = "round";
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    };
    if (d.s11_db_openems) drawCurve(d.s11_db_openems, C_OEMS);
    drawCurve(d.s11_db_gradenna, C_GRAD);

    // dip markers with offset labels (left/right to avoid collision)
    const annotate = (ys: number[], color: string, name: string, side: -1 | 1) => {
      let k = 0;
      ys.forEach((v, i) => { if (v < ys[k]) k = i; });
      const x = toX(d.freq_hz[k]);
      const y = toY(Math.max(yMin, ys[k]));
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.45;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + ch);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      const tx = x + side * 86;
      const ty = y + 30;
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.beginPath();
      ctx.moveTo(x, y + 5);
      ctx.lineTo(tx, ty - 9);
      ctx.stroke();
      const label = `${name} ${(d.freq_hz[k] / 1e9).toFixed(3)} GHz · ${ys[k].toFixed(1)} dB`;
      ctx.font = `10px ${T.mono}`;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(13,17,26,0.92)";
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(tx - tw / 2 - 7, ty - 9, tw + 14, 18, 5);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, tx, ty);
    };
    annotate(d.s11_db_gradenna, C_GRAD, "gradenna", -1);
    if (d.s11_db_openems) annotate(d.s11_db_openems, C_OEMS, "openEMS", 1);

    // legend (top right)
    const legend: [string, string][] = [["gradenna", C_GRAD]];
    if (d.s11_db_openems) legend.push(["openEMS", C_OEMS]);
    let lx = pad.left + cw - 10;
    ctx.font = `11px ${T.sans}`;
    ctx.textBaseline = "middle";
    for (const [name, color] of [...legend].reverse()) {
      ctx.textAlign = "right";
      ctx.fillStyle = T.dim;
      ctx.fillText(name, lx, pad.top + 12);
      const tw = ctx.measureText(name).width;
      ctx.fillStyle = color;
      ctx.fillRect(lx - tw - 26, pad.top + 10, 18, 4);
      lx -= tw + 44;
    }

    // hover crosshair + tooltip
    if (this.hoverX !== null && this.hoverX > pad.left && this.hoverX < pad.left + cw) {
      const f = fMin + ((this.hoverX - pad.left) / cw) * (fMax - fMin);
      let k = 0;
      let best = Infinity;
      d.freq_hz.forEach((ff, i) => {
        const e = Math.abs(ff - f);
        if (e < best) { best = e; k = i; }
      });
      const x = toX(d.freq_hz[k]);
      ctx.strokeStyle = "rgba(255,255,255,0.16)";
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + ch);
      ctx.stroke();

      const lines = [
        `${(d.freq_hz[k] / 1e9).toFixed(3)} GHz`,
        `gradenna ${d.s11_db_gradenna[k].toFixed(1)} dB`,
      ];
      if (d.s11_db_openems) lines.push(`openEMS ${d.s11_db_openems[k].toFixed(1)} dB`);
      ctx.font = `10px ${T.mono}`;
      const bw = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16;
      const bh = lines.length * 15 + 10;
      const bx = Math.min(x + 10, pad.left + cw - bw - 4);
      const by = pad.top + 8;
      ctx.fillStyle = "rgba(13,17,26,0.94)";
      ctx.strokeStyle = T.panelEdge;
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, 6);
      ctx.fill();
      ctx.stroke();
      ctx.textAlign = "left";
      lines.forEach((l, i) => {
        ctx.fillStyle = i === 0 ? T.text : i === 1 ? C_GRAD : C_OEMS;
        ctx.fillText(l, bx + 8, by + 14 + i * 15);
      });

      const series: [number[] | undefined, string][] = [
        [d.s11_db_gradenna, C_GRAD],
        [d.s11_db_openems, C_OEMS],
      ];
      for (const [ys, color] of series) {
        if (!ys) continue;
        ctx.beginPath();
        ctx.arc(x, toY(Math.max(yMin, ys[k])), 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }
  }

  dispose(): void {
    if (this.ro) this.ro.disconnect();
  }
}
