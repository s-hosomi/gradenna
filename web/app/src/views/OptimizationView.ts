import * as THREE from "three";
import { OptimizationData } from "../types";
import { COPPER, drawColorbar, lutToRGBA } from "../colormaps";
import { T, el, panel, sectionTitle, button, statRow, overlayMessage, hidpiCtx } from "../ui";

// Bicubic (Catmull-Rom) sampling of the density texture + LUT mapping +
// cross-fade between two consecutive frames, all on the GPU. This is what
// turns a 26x26 design grid into a smooth, print-quality density field.
const FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D frameA;
  uniform sampler2D frameB;
  uniform sampler2D lut;
  uniform float mixAB;
  uniform vec2 texel;     // 1/(ny,nx)
  uniform vec2 size;      // (ny,nx)

  float w0(float a){ return (1.0/6.0)*(a*(a*(-a+3.0)-3.0)+1.0); }
  float w1(float a){ return (1.0/6.0)*(a*a*(3.0*a-6.0)+4.0); }
  float w2(float a){ return (1.0/6.0)*(a*(a*(-3.0*a+3.0)+3.0)+1.0); }
  float w3(float a){ return (1.0/6.0)*(a*a*a); }

  float bicubic(sampler2D tex, vec2 uv) {
    vec2 st = uv * size - 0.5;
    vec2 base = floor(st);
    vec2 f = st - base;
    float result = 0.0;
    for (int j = -1; j <= 2; j++) {
      float wy = (j==-1) ? w0(f.y) : (j==0) ? w1(f.y) : (j==1) ? w2(f.y) : w3(f.y);
      for (int i = -1; i <= 2; i++) {
        float wx = (i==-1) ? w0(f.x) : (i==0) ? w1(f.x) : (i==1) ? w2(f.x) : w3(f.x);
        vec2 p = (base + vec2(float(i), float(j)) + 0.5) * texel;
        p = clamp(p, texel * 0.5, 1.0 - texel * 0.5);
        result += wx * wy * texture2D(tex, p).r;
      }
    }
    return result;
  }

  void main() {
    float a = bicubic(frameA, vUv);
    float b = bicubic(frameB, vUv);
    float d = clamp(mix(a, b, mixAB), 0.0, 1.0);
    vec3 c = texture2D(lut, vec2(d, 0.5)).rgb;
    // faint vignette to seat the field in the panel
    float r = distance(vUv, vec2(0.5));
    c *= 1.0 - 0.18 * smoothstep(0.45, 0.8, r);
    gl_FragColor = vec4(c, 1.0);
  }
`;

const VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export class OptimizationView {
  private container: HTMLElement;
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-0.5, 0.5, 0.5, -0.5, 0.1, 10);
  private renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  private texA: THREE.DataTexture | null = null;
  private texB: THREE.DataTexture | null = null;
  private material: THREE.ShaderMaterial | null = null;
  private mesh: THREE.Mesh | null = null;
  private chartCanvas: HTMLCanvasElement | null = null;
  private barCanvas: HTMLCanvasElement | null = null;
  private data: OptimizationData | null = null;
  private frame = 0; // fractional frame position during playback
  private playing = false;
  private rafId: number | null = null;
  private ro: ResizeObserver | null = null;
  private speed = 6; // frames per second of data
  private lastT = 0;

  constructor(container: HTMLElement) {
    this.container = container;
    this.camera.position.z = 1;
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
  }

  async init(): Promise<void> {
    const c = this.container;
    c.innerHTML = "";
    c.style.cssText =
      "display:flex;flex-direction:column;height:100%;gap:12px;padding:18px 20px;box-sizing:border-box;";

    const head = sectionTitle("Topology optimization", "an antenna grows by gradient descent");
    c.appendChild(head);

    const row = el("div", "display:flex;gap:14px;flex:1;min-height:0;");
    c.appendChild(row);

    // --- left: field panel + transport bar -------------------------------
    const left = el("div", "flex:1;min-width:0;display:flex;flex-direction:column;gap:10px;");
    row.appendChild(left);

    const fieldPanel = panel("flex:1;min-height:0;overflow:hidden;position:relative;");
    left.appendChild(fieldPanel);
    fieldPanel.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.cssText = "display:block;width:100%;height:100%;";

    const transport = panel(
      "display:flex;align-items:center;gap:12px;padding:8px 12px;flex-shrink:0;"
    );
    left.appendChild(transport);

    const playBtn = button("▶ Play", { primary: true });
    transport.appendChild(playBtn);

    const frameLabel = el(
      "span",
      `color:${T.dim};font-size:0.74rem;font-family:${T.mono};min-width:88px;`
    );
    transport.appendChild(frameLabel);

    const slider = el("input") as HTMLInputElement;
    slider.type = "range";
    slider.min = "0";
    slider.max = "0";
    slider.step = "0.01";
    slider.style.cssText = `flex:1;accent-color:${T.copper};height:4px;`;
    transport.appendChild(slider);

    const speedLabel = el("span", `color:${T.faint};font-size:0.72rem;font-family:${T.sans};`, "speed");
    transport.appendChild(speedLabel);
    const speedSel = el("select") as HTMLSelectElement;
    speedSel.style.cssText =
      `background:rgba(255,255,255,0.05);color:${T.text};border:1px solid ${T.panelEdge};` +
      `border-radius:6px;padding:3px 6px;font-size:0.74rem;font-family:${T.sans};`;
    for (const [label, v] of [["0.5×", 3], ["1×", 6], ["2×", 12]] as [string, number][]) {
      const o = el("option", "", label) as HTMLOptionElement;
      o.value = String(v);
      if (v === 6) o.selected = true;
      speedSel.appendChild(o);
    }
    transport.appendChild(speedSel);

    // --- right: chart + colorbar + stats ----------------------------------
    const right = el(
      "div",
      "width:300px;flex-shrink:0;display:flex;flex-direction:column;gap:10px;min-height:0;"
    );
    row.appendChild(right);

    const chartPanel = panel("flex:1;min-height:140px;padding:10px;display:flex;flex-direction:column;gap:6px;");
    right.appendChild(chartPanel);
    chartPanel.appendChild(
      el("div", `color:${T.dim};font-size:0.74rem;font-family:${T.sans};`, "Objective convergence (log scale)")
    );
    this.chartCanvas = el("canvas", "width:100%;flex:1;display:block;") as HTMLCanvasElement;
    chartPanel.appendChild(this.chartCanvas);

    const infoPanel = panel("padding:10px 12px;display:flex;gap:12px;align-items:stretch;");
    right.appendChild(infoPanel);
    this.barCanvas = el("canvas", "width:56px;height:150px;flex-shrink:0;") as HTMLCanvasElement;
    infoPanel.appendChild(this.barCanvas);
    const stats = el("div", "flex:1;display:flex;flex-direction:column;justify-content:center;gap:2px;");
    infoPanel.appendChild(stats);
    const sIter = statRow("iteration");
    const sObj = statRow("objective");
    const sGain = statRow("vs. start");
    const sExtent = statRow("design region");
    for (const s of [sIter, sObj, sGain, sExtent]) stats.appendChild(s.row);

    // --- load data ---------------------------------------------------------
    try {
      const resp = await fetch("./data/optimization.json");
      if (!resp.ok) throw new Error(String(resp.status));
      this.data = (await resp.json()) as OptimizationData;
    } catch {
      overlayMessage(
        fieldPanel,
        `No data yet —<br><code style="color:${T.copper}">python scripts/export_viz.py</code>`
      );
      return;
    }
    const d = this.data;
    const nFrames = d.frames.length;

    const makeTex = () => {
      const t = new THREE.DataTexture(
        new Uint8Array(new ArrayBuffer(d.nx * d.ny)),
        d.ny,
        d.nx,
        THREE.RedFormat,
        THREE.UnsignedByteType
      );
      t.needsUpdate = true;
      return t;
    };
    this.texA = makeTex();
    this.texB = makeTex();
    const lutTex = new THREE.DataTexture(lutToRGBA(COPPER), 256, 1, THREE.RGBAFormat);
    lutTex.needsUpdate = true;

    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        frameA: { value: this.texA },
        frameB: { value: this.texB },
        lut: { value: lutTex },
        mixAB: { value: 0 },
        texel: { value: new THREE.Vector2(1 / d.ny, 1 / d.nx) },
        size: { value: new THREE.Vector2(d.ny, d.nx) },
      },
    });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.material);
    this.scene.add(this.mesh);

    slider.max = String(nFrames - 1);
    sExtent.value.textContent = `${d.extent_mm[0]}×${d.extent_mm[1]} mm`;
    drawColorbar(
      this.barCanvas,
      COPPER,
      [
        { t: 0, label: "air" },
        { t: 0.5, label: "0.5" },
        { t: 1, label: "copper" },
      ],
      "ρ"
    );

    const uploadFrame = (tex: THREE.DataTexture, idx: number) => {
      const src = d.frames[Math.max(0, Math.min(nFrames - 1, idx))];
      const buf = tex.image.data as Uint8Array;
      for (let k = 0; k < src.length; k++) buf[k] = Math.round(src[k] * 255);
      tex.needsUpdate = true;
    };

    let loadedA = -1;
    let loadedB = -1;
    const showFrame = (f: number) => {
      const i0 = Math.floor(f);
      const i1 = Math.min(nFrames - 1, i0 + 1);
      if (loadedA !== i0) { uploadFrame(this.texA!, i0); loadedA = i0; }
      if (loadedB !== i1) { uploadFrame(this.texB!, i1); loadedB = i1; }
      this.material!.uniforms.mixAB.value = f - i0;
      this.renderer.render(this.scene, this.camera);

      const idx = Math.round(f);
      frameLabel.textContent = `iter ${String(idx).padStart(2, "0")} / ${nFrames - 1}`;
      sIter.value.textContent = String(idx);
      sObj.value.textContent = d.objective[idx].toExponential(2);
      const gain = d.objective[idx] / Math.max(d.objective[0], 1e-30);
      sGain.value.textContent = `${gain >= 100 ? gain.toFixed(0) : gain.toFixed(1)}×`;
      this.drawChart(d, f);
    };

    // --- interactions ------------------------------------------------------
    playBtn.addEventListener("click", () => {
      this.playing = !this.playing;
      playBtn.textContent = this.playing ? "❚❚ Pause" : "▶ Play";
      if (this.playing) {
        if (this.frame >= nFrames - 1) this.frame = 0;
        this.lastT = performance.now();
        const tick = (t: number) => {
          if (!this.playing) return;
          this.frame += ((t - this.lastT) / 1000) * this.speed;
          this.lastT = t;
          if (this.frame >= nFrames - 1) {
            this.frame = nFrames - 1;
            this.playing = false;
            playBtn.textContent = "▶ Replay";
          }
          slider.value = String(this.frame);
          showFrame(this.frame);
          this.rafId = requestAnimationFrame(tick);
        };
        this.rafId = requestAnimationFrame(tick);
      }
    });

    slider.addEventListener("input", () => {
      this.playing = false;
      playBtn.textContent = "▶ Play";
      this.frame = parseFloat(slider.value);
      showFrame(this.frame);
    });

    speedSel.addEventListener("change", () => (this.speed = parseFloat(speedSel.value)));

    this.ro = new ResizeObserver(() => {
      const w = fieldPanel.clientWidth;
      const h = fieldPanel.clientHeight;
      if (w < 2 || h < 2) return;
      this.renderer.setSize(w, h);
      // keep the design region square and centered
      const aspect = w / h;
      if (aspect >= 1) {
        this.camera.left = -aspect / 2;
        this.camera.right = aspect / 2;
        this.camera.top = 0.5;
        this.camera.bottom = -0.5;
      } else {
        this.camera.left = -0.5;
        this.camera.right = 0.5;
        this.camera.top = 0.5 / aspect;
        this.camera.bottom = -0.5 / aspect;
      }
      this.camera.updateProjectionMatrix();
      showFrame(this.frame);
    });
    this.ro.observe(fieldPanel);

    showFrame(0);
  }

  private drawChart(d: OptimizationData, frame: number): void {
    if (!this.chartCanvas) return;
    const res = hidpiCtx(this.chartCanvas);
    if (!res) return;
    const { ctx, w, h } = res;

    const pad = { top: 8, right: 10, bottom: 26, left: 46 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;
    const vals = d.objective.map((v) => Math.log10(Math.max(v, 1e-30)));
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const range = maxV - minV || 1;
    const toX = (i: number) => pad.left + (i / (vals.length - 1)) * cw;
    const toY = (v: number) => pad.top + ch - ((v - minV) / range) * ch;

    ctx.clearRect(0, 0, w, h);

    // gridlines at integer decades
    ctx.font = `9.5px ${T.mono}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let dec = Math.ceil(minV); dec <= Math.floor(maxV); dec++) {
      const y = toY(dec);
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + cw, y);
      ctx.stroke();
      ctx.fillStyle = T.faint;
      ctx.fillText(`1e${dec}`, pad.left - 6, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = T.faint;
    ctx.fillText("iteration", pad.left + cw / 2, h - 8);

    // area fill under the curve
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
    grad.addColorStop(0, "rgba(232,148,60,0.28)");
    grad.addColorStop(1, "rgba(232,148,60,0.02)");
    ctx.beginPath();
    vals.forEach((v, i) => (i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v))));
    ctx.lineTo(toX(vals.length - 1), pad.top + ch);
    ctx.lineTo(toX(0), pad.top + ch);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // curve
    ctx.beginPath();
    vals.forEach((v, i) => (i === 0 ? ctx.moveTo(toX(i), toY(v)) : ctx.lineTo(toX(i), toY(v))));
    ctx.strokeStyle = T.copper;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = "round";
    ctx.stroke();

    // current position
    const fi = Math.max(0, Math.min(vals.length - 1, frame));
    const i0 = Math.floor(fi);
    const v = vals[i0] + (vals[Math.min(vals.length - 1, i0 + 1)] - vals[i0]) * (fi - i0);
    const x = toX(fi);
    const y = toY(v);
    ctx.strokeStyle = "rgba(255,255,255,0.14)";
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(x, pad.top);
    ctx.lineTo(x, pad.top + ch);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  dispose(): void {
    this.playing = false;
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    if (this.ro) this.ro.disconnect();
    this.texA?.dispose();
    this.texB?.dispose();
    this.material?.dispose();
    this.mesh?.geometry.dispose();
    this.renderer.dispose();
  }
}
