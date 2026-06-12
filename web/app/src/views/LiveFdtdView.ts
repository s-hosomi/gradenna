import * as THREE from "three";
import { fillDivergingRGBA, COPPER, DARK_DIV, VIRIDIS, lutColor, drawColorbar } from "../colormaps";
import { Fdtd2DInstance, GradennaWasmModule, OptimizationData } from "../types";
import { T, el, panel, sectionTitle, button, styleButton, statRow } from "../ui";

const NX = 256;
const NY = 256;
const DX_M = 0.001; // 1 mm cells -> 256 x 256 mm domain, lambda(2.4 GHz) = 125 cells
const NPML = 10;
const TWO_PI = 2 * Math.PI;
const SIGMA_METAL = 1e7;

type Brush = "source" | "draw" | "erase";
type Excitation = "pulse" | "cw";
type Display = "ez" | "intensity";

export class LiveFdtdView {
  private container: HTMLElement;
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0.1, 10);
  private renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
  private texture: THREE.DataTexture | null = null;
  private rgba = new Uint8Array(new ArrayBuffer(NX * NY * 4));
  private rafId: number | null = null;
  private ro: ResizeObserver | null = null;

  private fdtd: Fdtd2DInstance | null = null;
  private wasmMemory: WebAssembly.Memory | null = null;
  private time = 0;
  private stepCount = 0;
  private sourceI = Math.round(NX * 0.5);
  private sourceJ = Math.round(NY * 0.3);
  private sigma = new Float32Array(NX * NY);
  private brush: Brush = "source";
  private brushSize = 5;
  private excitation: Excitation = "cw";
  private display: Display = "ez";
  private stepsPerFrame = 8;
  private ezAmplitude = 1e-6;
  private dragging = false;
  // time-averaged |Ez|^2 (the "exposure" picture: interference fringes,
  // radiation patterns) accumulated since the last restart
  private intensity = new Float32Array(NX * NY);
  private optData: OptimizationData | null = null;

  // excitation frequency is per-scene: the double-slit needs a shorter
  // wavelength than the antenna scenes to fit fringes in a 256 mm domain
  private fc = 2.4e9;
  private readonly BW = 1.2e9;
  private dt = 0;
  private onFreqChange: (() => void) | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.camera.position.z = 1;
    this.renderer.setPixelRatio(1);
  }

  async init(): Promise<void> {
    const c = this.container;
    c.innerHTML = "";
    c.style.cssText =
      "display:flex;flex-direction:column;height:100%;gap:12px;padding:18px 20px;box-sizing:border-box;";
    c.appendChild(
      sectionTitle("Live FDTD", "the wasm kernel solving Maxwell in your browser")
    );

    // --- wasm ---------------------------------------------------------------
    let mod: GradennaWasmModule | null = null;
    try {
      mod = (await import(/* @vite-ignore */ "@wasm/gradenna_wasm.js")) as GradennaWasmModule;
      const initOut = await mod.default();
      this.wasmMemory = (initOut && initOut.memory) || mod.memory || null;
      this.fdtd = new mod.Fdtd2D(NX, NY, DX_M, NPML);
      this.dt = this.fdtd.dt_seconds();
    } catch {
      const info = el(
        "div",
        `flex:1;display:flex;align-items:center;justify-content:center;color:${T.dim};` +
          `text-align:center;line-height:2;font-family:${T.sans};`
      );
      info.innerHTML =
        `<div><div style="font-size:1.05rem;margin-bottom:10px;color:${T.text};">wasm kernel not built</div>` +
        `<code style="display:block;margin:10px auto;padding:10px 16px;background:rgba(255,255,255,0.04);` +
        `border:1px solid ${T.panelEdge};border-radius:8px;font-size:0.82rem;color:${T.copper};">` +
        `cd web/wasm-kernel && wasm-pack build --target web --out-dir pkg</code>` +
        `<div style="color:${T.faint};font-size:0.78rem;">the other tabs work without it</div></div>`;
      c.appendChild(info);
      return;
    }

    // the optimized design for the "Antenna" scene (optional, never fatal)
    try {
      const r = await fetch("./data/optimization.json");
      if (r.ok) this.optData = (await r.json()) as OptimizationData;
    } catch { /* scene button will report */ }

    // --- layout --------------------------------------------------------------
    const row = el("div", "display:flex;gap:14px;flex:1;min-height:0;");
    c.appendChild(row);

    const fieldPanel = panel(
      "flex:1;min-width:0;overflow:hidden;position:relative;cursor:crosshair;"
    );
    row.appendChild(fieldPanel);
    fieldPanel.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.cssText = "display:block;width:100%;height:100%;";

    const side = el(
      "div",
      "width:230px;flex-shrink:0;display:flex;flex-direction:column;gap:10px;min-height:0;overflow-y:auto;"
    );
    row.appendChild(side);

    const caption = (text: string) =>
      el("div", `color:${T.faint};font-size:0.7rem;font-family:${T.sans};text-transform:uppercase;letter-spacing:0.08em;`, text);

    // tools
    const toolPanel = panel("padding:10px 12px;display:flex;flex-direction:column;gap:6px;");
    side.appendChild(toolPanel);
    toolPanel.appendChild(caption("tool"));
    const toolRow = el("div", "display:flex;gap:6px;flex-wrap:wrap;");
    toolPanel.appendChild(toolRow);
    const toolBtns = new Map<Brush, HTMLButtonElement>();
    for (const [mode, label] of [["source", "⊕ Source"], ["draw", "✏ Copper"], ["erase", "⌫ Erase"]] as [Brush, string][]) {
      const b = button(label, { active: mode === this.brush });
      b.addEventListener("click", () => {
        this.brush = mode;
        toolBtns.forEach((bb, m) => styleButton(bb, { active: m === mode }));
        fieldPanel.style.cursor = mode === "source" ? "crosshair" : "cell";
      });
      toolBtns.set(mode, b);
      toolRow.appendChild(b);
    }
    const brushRow = el("div", "display:flex;align-items:center;gap:8px;margin-top:4px;");
    toolPanel.appendChild(brushRow);
    brushRow.appendChild(el("span", `color:${T.faint};font-size:0.72rem;font-family:${T.sans};`, "brush"));
    const brushSlider = el("input") as HTMLInputElement;
    brushSlider.type = "range";
    brushSlider.min = "2";
    brushSlider.max = "16";
    brushSlider.value = String(this.brushSize);
    brushSlider.style.cssText = `flex:1;accent-color:${T.copper};height:4px;`;
    brushSlider.addEventListener("input", () => (this.brushSize = parseInt(brushSlider.value, 10)));
    brushRow.appendChild(brushSlider);

    // excitation
    const excPanel = panel("padding:10px 12px;display:flex;flex-direction:column;gap:6px;");
    side.appendChild(excPanel);
    excPanel.appendChild(caption("excitation"));
    const excRow = el("div", "display:flex;gap:6px;");
    excPanel.appendChild(excRow);
    const excBtns = new Map<Excitation, HTMLButtonElement>();
    for (const [mode, label] of [["cw", "∿ CW"], ["pulse", "⌒ Pulse"]] as [Excitation, string][]) {
      const b = button(label, { active: mode === this.excitation });
      b.addEventListener("click", () => {
        this.excitation = mode;
        this.time = 0;
        excBtns.forEach((bb, m) => styleButton(bb, { active: m === mode }));
      });
      excBtns.set(mode, b);
      excRow.appendChild(b);
    }
    const speedRow = el("div", "display:flex;align-items:center;gap:8px;margin-top:2px;");
    excPanel.appendChild(speedRow);
    speedRow.appendChild(el("span", `color:${T.faint};font-size:0.72rem;font-family:${T.sans};`, "speed"));
    const speedSlider = el("input") as HTMLInputElement;
    speedSlider.type = "range";
    speedSlider.min = "1";
    speedSlider.max = "16";
    speedSlider.value = String(this.stepsPerFrame);
    speedSlider.style.cssText = `flex:1;accent-color:${T.accent};height:4px;`;
    speedSlider.addEventListener("input", () => (this.stepsPerFrame = parseInt(speedSlider.value, 10)));
    speedRow.appendChild(speedSlider);

    // display mode
    const dispPanel = panel("padding:10px 12px;display:flex;flex-direction:column;gap:6px;");
    side.appendChild(dispPanel);
    dispPanel.appendChild(caption("display"));
    const dispRow = el("div", "display:flex;gap:6px;");
    dispPanel.appendChild(dispRow);
    const dispBtns = new Map<Display, HTMLButtonElement>();

    // colorbar (declared early so the display toggle can redraw it)
    const bar = el("canvas", "width:52px;height:120px;flex-shrink:0;") as HTMLCanvasElement;
    const drawBar = () => {
      if (this.display === "ez") {
        drawColorbar(bar, DARK_DIV, [
          { t: 0, label: "−" },
          { t: 0.5, label: "0" },
          { t: 1, label: "+" },
        ], "Ez");
      } else {
        drawColorbar(bar, VIRIDIS, [
          { t: 0, label: "0" },
          { t: 1, label: "max" },
        ], "⟨Ez²⟩");
      }
    };
    for (const [mode, label] of [["ez", "Ez field"], ["intensity", "Intensity"]] as [Display, string][]) {
      const b = button(label, { active: mode === this.display });
      b.addEventListener("click", () => {
        this.display = mode;
        if (mode === "intensity") this.intensity.fill(0); // fresh exposure
        dispBtns.forEach((bb, m) => styleButton(bb, { active: m === mode }));
        drawBar();
      });
      dispBtns.set(mode, b);
      dispRow.appendChild(b);
    }
    dispPanel.appendChild(
      el("div", `color:${T.faint};font-size:0.7rem;font-family:${T.sans};line-height:1.5;`,
        "Intensity = time-averaged Ez² — fringes and patterns emerge")
    );

    // scenes
    const scenePanel = panel("padding:10px 12px;display:flex;flex-direction:column;gap:6px;");
    side.appendChild(scenePanel);
    scenePanel.appendChild(caption("scenes"));
    const sceneRow = el("div", "display:flex;gap:6px;flex-wrap:wrap;");
    scenePanel.appendChild(sceneRow);
    const antBtn = button("Antenna");
    const slitBtn = button("Double slit");
    const mirrorBtn = button("Mirror");
    const clearBtn = button("Clear");
    sceneRow.append(antBtn, slitBtn, mirrorBtn, clearBtn);
    antBtn.addEventListener("click", () => this.presetAntenna());
    slitBtn.addEventListener("click", () => this.presetDoubleSlit());
    mirrorBtn.addEventListener("click", () => this.presetMirror());
    clearBtn.addEventListener("click", () => this.clearAll());
    if (!this.optData) {
      antBtn.disabled = true;
      antBtn.style.opacity = "0.4";
      antBtn.title = "needs data/optimization.json";
    } else {
      antBtn.title = "load the optimized design from the Optimization tab";
    }

    const hud = panel("padding:10px 12px;display:flex;gap:12px;");
    side.appendChild(hud);
    hud.appendChild(bar);
    const stats = el("div", "flex:1;display:flex;flex-direction:column;justify-content:center;gap:2px;");
    hud.appendChild(stats);
    const sFreq = statRow("f");
    const sTime = statRow("t");
    const sStep = statRow("step");
    const sGrid = statRow("grid");
    const sCell = statRow("cell");
    for (const s of [sFreq, sTime, sStep, sGrid, sCell]) stats.appendChild(s.row);
    const showFreq = () => (sFreq.value.textContent = `${(this.fc / 1e9).toFixed(1)} GHz`);
    showFreq();
    this.onFreqChange = showFreq;
    sGrid.value.textContent = `${NX}×${NY}`;
    sCell.value.textContent = `${DX_M * 1e3} mm`;
    drawBar();

    const hint = el(
      "div",
      `color:${T.faint};font-size:0.72rem;line-height:1.7;font-family:${T.sans};padding:0 2px;`
    );
    hint.innerHTML =
      "Click to move the source, paint<br>copper to reflect and diffract.<br>Try <b>Antenna</b> + <b>Intensity</b>: the<br>optimized design radiating live.";
    side.appendChild(hint);

    // --- texture / scene -------------------------------------------------------
    this.texture = new THREE.DataTexture(this.rgba, NY, NX, THREE.RGBAFormat);
    this.texture.flipY = false;
    // sRGB-authored bytes; declaring it keeps the dark background dark
    this.texture.colorSpace = THREE.SRGBColorSpace;
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: this.texture })
    );
    this.scene.add(mesh);

    // --- pointer interaction ------------------------------------------------------
    const gridCoords = (e: PointerEvent): [number, number] => {
      const rect = fieldPanel.getBoundingClientRect();
      const v = (e.clientY - rect.top) / rect.height;
      const u = (e.clientX - rect.left) / rect.width;
      return [Math.floor(v * NX), Math.floor(u * NY)];
    };
    const applyAt = (e: PointerEvent) => {
      const [gi, gj] = gridCoords(e);
      if (this.brush === "source") {
        this.sourceI = Math.max(NPML + 2, Math.min(NX - NPML - 3, gi));
        this.sourceJ = Math.max(NPML + 2, Math.min(NY - NPML - 3, gj));
        this.intensity.fill(0);
      } else {
        this.paint(gi, gj, this.brushSize, this.brush === "draw" ? SIGMA_METAL : 0);
      }
    };
    fieldPanel.addEventListener("pointerdown", (e: PointerEvent) => {
      this.dragging = true;
      fieldPanel.setPointerCapture(e.pointerId);
      applyAt(e);
    });
    fieldPanel.addEventListener("pointermove", (e: PointerEvent) => {
      if (this.dragging && this.brush !== "source") applyAt(e);
    });
    fieldPanel.addEventListener("pointerup", () => (this.dragging = false));

    this.ro = new ResizeObserver(() => {
      const w = fieldPanel.clientWidth;
      const h = fieldPanel.clientHeight;
      if (w < 2 || h < 2) return;
      this.renderer.setSize(w, h);
      const aspect = w / h;
      this.camera.left = -aspect / 2;
      this.camera.right = aspect / 2;
      this.camera.top = 0.5;
      this.camera.bottom = -0.5;
      this.camera.updateProjectionMatrix();
    });
    this.ro.observe(fieldPanel);

    // --- main loop -------------------------------------------------------------------
    const animate = () => {
      this.rafId = requestAnimationFrame(animate);
      const f = this.fdtd!;
      for (let s = 0; s < this.stepsPerFrame; s++) {
        f.step(this.sourceI, this.sourceJ, this.sourceValue(this.time));
        this.time += this.dt;
        this.stepCount++;
      }
      sTime.value.textContent = `${(this.time * 1e9).toFixed(2)} ns`;
      sStep.value.textContent = String(this.stepCount);

      let ez: Float32Array;
      if (this.wasmMemory) {
        ez = new Float32Array(this.wasmMemory.buffer, f.ez_ptr(), NX * NY);
      } else {
        ez = new Float32Array(NX * NY);
      }

      let maxAbs = 0;
      for (let k = 0; k < ez.length; k++) {
        const a = Math.abs(ez[k]);
        if (a > maxAbs) maxAbs = a;
        this.intensity[k] += ez[k] * ez[k];
      }
      this.ezAmplitude = maxAbs > this.ezAmplitude
        ? this.ezAmplitude * 0.7 + maxAbs * 0.3
        : this.ezAmplitude * 0.985 + maxAbs * 0.015;

      if (this.display === "ez") {
        fillDivergingRGBA(this.rgba, ez, NX * NY, Math.max(this.ezAmplitude, 1e-9), DARK_DIV, 0.45);
      } else {
        this.fillIntensity();
      }
      this.compositeOverlays();
      this.texture!.needsUpdate = true;
      this.renderer.render(this.scene, this.camera);
    };
    animate();
  }

  /** Time-averaged Ez² with a sqrt tone curve (≈|E| amplitude) in viridis. */
  private fillIntensity(): void {
    let max = 0;
    for (let k = 0; k < this.intensity.length; k++) {
      if (this.intensity[k] > max) max = this.intensity[k];
    }
    const inv = max > 0 ? 1 / max : 0;
    // gain x2.6: the source cell dominates the max, so an unscaled sqrt would
    // leave the radiated pattern in the bottom of the colormap
    for (let k = 0; k < this.intensity.length; k++) {
      const t = 2.6 * Math.sqrt(this.intensity[k] * inv);
      const idx = Math.round(Math.min(1, t) * 255) * 3;
      const p = k * 4;
      this.rgba[p] = VIRIDIS[idx];
      this.rgba[p + 1] = VIRIDIS[idx + 1];
      this.rgba[p + 2] = VIRIDIS[idx + 2];
      this.rgba[p + 3] = 255;
    }
  }

  /** Copper cells, CPML shading and the source marker, drawn over the field. */
  private compositeOverlays(): void {
    const [cr, cg, cb] = lutColor(COPPER, 0.85);
    const [hr, hg, hb] = lutColor(COPPER, 1.0);
    for (let i = 0; i < NX; i++) {
      const inPmlI = i < NPML || i >= NX - NPML;
      for (let j = 0; j < NY; j++) {
        const k = i * NY + j;
        const p = k * 4;
        if (this.sigma[k] > 0) {
          const edge =
            (i > 0 && this.sigma[k - NY] === 0) ||
            (i < NX - 1 && this.sigma[k + NY] === 0) ||
            (j > 0 && this.sigma[k - 1] === 0) ||
            (j < NY - 1 && this.sigma[k + 1] === 0);
          this.rgba[p] = edge ? hr : cr;
          this.rgba[p + 1] = edge ? hg : cg;
          this.rgba[p + 2] = edge ? hb : cb;
        } else if (inPmlI || j < NPML || j >= NY - NPML) {
          this.rgba[p] = (this.rgba[p] * 90) >> 8;
          this.rgba[p + 1] = (this.rgba[p + 1] * 90) >> 8;
          this.rgba[p + 2] = (this.rgba[p + 2] * 95) >> 8;
        }
      }
    }
    const R = 5;
    for (let a = 0; a < 64; a++) {
      const ang = (a / 64) * TWO_PI;
      const i = Math.round(this.sourceI + R * Math.sin(ang));
      const j = Math.round(this.sourceJ + R * Math.cos(ang));
      if (i < 0 || i >= NX || j < 0 || j >= NY) continue;
      const p = (i * NY + j) * 4;
      this.rgba[p] = 255;
      this.rgba[p + 1] = 255;
      this.rgba[p + 2] = 255;
    }
  }

  private sourceValue(t: number): number {
    if (this.excitation === "cw") {
      const ramp = Math.min(1, t / (4 / this.fc + 1e-30));
      return ramp * Math.sin(TWO_PI * this.fc * t);
    }
    const tau = 1 / (TWO_PI * this.BW);
    const t0 = 4 * tau * TWO_PI;
    const env = Math.exp(-((t - t0) ** 2) / (2 * tau * tau));
    return env * Math.sin(TWO_PI * this.fc * t);
  }

  private paint(ci: number, cj: number, radius: number, value: number): void {
    for (let di = -radius; di <= radius; di++) {
      for (let dj = -radius; dj <= radius; dj++) {
        if (di * di + dj * dj > radius * radius) continue;
        const i = ci + di;
        const j = cj + dj;
        if (i < NPML || i >= NX - NPML || j < NPML || j >= NY - NPML) continue;
        this.sigma[i * NY + j] = value;
      }
    }
    this.fdtd?.set_sigma(this.sigma);
  }

  /** The optimized design from the Optimization tab, radiating live. */
  private presetAntenna(): void {
    if (!this.optData) return;
    const d = this.optData;
    this.sigma.fill(0);
    const frame = d.frames[d.frames.length - 1];
    // honour the design's physical cell size: the optimization may have been
    // run at a different resolution than this view's 1 mm cells
    const designDxMm = d.extent_mm[0] / d.nx;
    const scale = designDxMm / (DX_M * 1e3);
    const sx = Math.round(d.nx * scale);
    const sy = Math.round(d.ny * scale);
    const oi = Math.round((NX - sx) / 2);
    const oj = Math.round((NY - sy) / 2);
    for (let i = 0; i < sx; i++) {
      const di = Math.min(d.nx - 1, Math.floor(i / scale));
      for (let j = 0; j < sy; j++) {
        const dj = Math.min(d.ny - 1, Math.floor(j / scale));
        const gi = oi + i;
        const gj = oj + j;
        if (gi < NPML || gi >= NX - NPML || gj < NPML || gj >= NY - NPML) continue;
        if (frame[di * d.ny + dj] > 0.5) this.sigma[gi * NY + gj] = SIGMA_METAL;
      }
    }
    // feed location: exported with the data when available, else the
    // canonical problem's feed (row-centre, ~1/5 in from the low-j edge)
    const fc = (d as unknown as { feed_cell?: [number, number] }).feed_cell;
    const [fi, fj] = fc ?? [Math.round(d.nx / 2), Math.round(d.ny / 5)];
    this.sourceI = Math.max(NPML + 2, Math.min(NX - NPML - 3, oi + Math.round(fi * scale)));
    this.sourceJ = Math.max(NPML + 2, Math.min(NY - NPML - 3, oj + Math.round(fj * scale)));
    // keep the feed cell itself non-metal (it is masked in the optimization)
    this.sigma[this.sourceI * NY + this.sourceJ] = 0;
    this.excitation = "cw";
    this.setFreq(2.4e9);
    this.fdtd?.set_sigma(this.sigma);
    this.restart();
  }

  private setFreq(f: number): void {
    this.fc = f;
    this.onFreqChange?.();
  }

  private presetDoubleSlit(): void {
    this.clearAll();
    // 4.8 GHz here: lambda = 62 cells, so several fringes fit behind the wall
    this.setFreq(4.8e9);
    const wallJ = Math.round(NY * 0.5);
    const slitHalf = 8;
    const sep = 56;
    const c1 = Math.round(NX / 2 - sep / 2);
    const c2 = Math.round(NX / 2 + sep / 2);
    for (let i = NPML; i < NX - NPML; i++) {
      if (Math.abs(i - c1) <= slitHalf || Math.abs(i - c2) <= slitHalf) continue;
      for (let w = 0; w < 4; w++) this.sigma[i * NY + wallJ + w] = SIGMA_METAL;
    }
    this.sourceI = Math.round(NX / 2);
    this.sourceJ = Math.round(NY * 0.25);
    this.fdtd?.set_sigma(this.sigma);
    this.restart();
  }

  private presetMirror(): void {
    this.clearAll();
    this.setFreq(2.4e9);
    const focusI = Math.round(NX / 2);
    const vertexJ = Math.round(NY * 0.18);
    const F = 55; // focal length in cells
    this.sourceI = focusI;
    this.sourceJ = vertexJ + F;
    for (let i = NPML; i < NX - NPML; i++) {
      const di = i - focusI;
      const j = Math.round(vertexJ + (di * di) / (4 * F));
      if (j < NPML || j >= NY - NPML) continue;
      for (let w = 0; w < 4; w++) {
        const jj = j - w;
        if (jj >= NPML) this.sigma[i * NY + jj] = SIGMA_METAL;
      }
    }
    this.fdtd?.set_sigma(this.sigma);
    this.restart();
  }

  private clearAll(): void {
    this.sigma.fill(0);
    this.setFreq(2.4e9);
    this.fdtd?.set_sigma(this.sigma);
    this.restart();
  }

  private restart(): void {
    this.fdtd?.reset();
    this.time = 0;
    this.stepCount = 0;
    this.ezAmplitude = 1e-6;
    this.intensity.fill(0);
  }

  dispose(): void {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    if (this.ro) this.ro.disconnect();
    this.texture?.dispose();
    this.renderer.dispose();
  }
}
