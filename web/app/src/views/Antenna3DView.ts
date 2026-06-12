import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { FarField3DData } from "../types";
import { VIRIDIS, lutColor, drawColorbar } from "../colormaps";
import { T, el, panel, sectionTitle, button, styleButton, statRow, overlayMessage } from "../ui";

// Near-field volume exported by scripts/export_viz.py (frozen contract).
export interface NearField3DData {
  kind: string;
  freq_hz: number;
  shape: [number, number, number];
  spacing_mm: [number, number, number];
  e_mag: number[]; // row-major idx = i*ny*nz + j*nz + k
  e_max: number;
  geometry: {
    board_mm: [number, number, number, number];
    patch_mm: [number, number, number, number];
    z_gnd_mm: number;
    z_patch_mm: number;
    h_sub_mm: number;
    feed_mm: [number, number];
    npml: number;
  };
}

// |E| -> [0,1] over DB_RANGE decades below the volume max (log scale: the
// near field spans orders of magnitude, linear would show only the feed).
const DB_DECADES = 3;

function fieldT(e: number, eMax: number): number {
  const r = Math.log10(Math.max(e, 1e-300) / eMax) / DB_DECADES + 1;
  return Math.max(0, Math.min(1, r));
}

function textSprite(text: string, color = "#9aa3b5", px = 26): THREE.Sprite {
  const pad = 8;
  const c = document.createElement("canvas");
  const m = c.getContext("2d")!;
  m.font = `${px}px ${T.sans}`;
  const w = Math.ceil(m.measureText(text).width) + pad * 2;
  const h = px + pad * 2;
  c.width = w * 2;
  c.height = h * 2;
  const m2 = c.getContext("2d")!;
  m2.scale(2, 2);
  m2.font = `${px}px ${T.sans}`;
  m2.fillStyle = color;
  m2.textBaseline = "middle";
  m2.fillText(text, pad, h / 2);
  const tex = new THREE.CanvasTexture(c);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false })
  );
  const scale = 0.0034 * px;
  sprite.scale.set((w / h) * scale, scale, 1);
  return sprite;
}

/**
 * The flagship 3D scene: the real benchmark patch geometry, the |E| near
 * field of the actual 3D simulation as draggable translucent slices, and the
 * far-field directivity lobe floating above it — geometry, near field and far
 * field of the same antenna in one picture.
 */
export class Antenna3DView {
  private container: HTMLElement;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  private renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  private controls: OrbitControls | null = null;
  private rafId: number | null = null;
  private ro: ResizeObserver | null = null;

  private ff: FarField3DData | null = null;
  private nf: NearField3DData | null = null;
  private lobe: THREE.Group | null = null;
  private sliceX: THREE.Mesh | null = null; // vertical (y-z) plane at x index
  private sliceZ: THREE.Mesh | null = null; // horizontal (x-y) plane at z index
  private sliceXTex: THREE.DataTexture | null = null;
  private sliceZTex: THREE.DataTexture | null = null;
  private useDb = false;
  private readonly DB_FLOOR = -30;

  // scene scale: mm -> scene units
  private s = 1 / 90;
  private center = new THREE.Vector3();

  constructor(container: HTMLElement) {
    this.container = container;
    this.camera.position.set(1.7, 1.25, 1.7);
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
  }

  async init(): Promise<void> {
    const c = this.container;
    c.innerHTML = "";
    c.style.cssText =
      "display:flex;flex-direction:column;height:100%;gap:12px;padding:18px 20px;box-sizing:border-box;";

    const head = el("div", "display:flex;align-items:center;gap:14px;");
    head.appendChild(
      sectionTitle("Antenna 3D", "geometry · near field · far field of the 2.45 GHz patch")
    );
    head.appendChild(el("div", "flex:1;"));
    const linBtn = button("Linear", { active: true });
    const dbBtn = button("dB");
    head.append(linBtn, dbBtn);
    c.appendChild(head);

    const row = el("div", "display:flex;gap:14px;flex:1;min-height:0;");
    c.appendChild(row);

    const viewPanel = panel("flex:1;min-width:0;overflow:hidden;position:relative;");
    viewPanel.style.background =
      `radial-gradient(ellipse 90% 70% at 50% 38%, #141a2a 0%, ${T.panel} 65%, #0d1018 100%)`;
    row.appendChild(viewPanel);
    viewPanel.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.cssText = "display:block;width:100%;height:100%;";

    const side = el(
      "div",
      "width:240px;flex-shrink:0;display:flex;flex-direction:column;gap:10px;min-height:0;overflow-y:auto;"
    );
    row.appendChild(side);

    // layers panel
    const layers = panel("padding:10px 12px;display:flex;flex-direction:column;gap:7px;");
    side.appendChild(layers);
    layers.appendChild(
      el("div", `color:${T.faint};font-size:0.7rem;font-family:${T.sans};text-transform:uppercase;letter-spacing:0.08em;`, "layers")
    );
    const mkCheck = (label: string, checked: boolean, onChange: (v: boolean) => void) => {
      const wrap = el("label", `display:flex;align-items:center;gap:8px;cursor:pointer;color:${T.text};font-size:0.78rem;font-family:${T.sans};`);
      const cb = el("input") as HTMLInputElement;
      cb.type = "checkbox";
      cb.checked = checked;
      cb.style.accentColor = T.copper;
      cb.addEventListener("change", () => onChange(cb.checked));
      wrap.append(cb, document.createTextNode(label));
      layers.appendChild(wrap);
      return cb;
    };

    // slice sliders
    const slicePanel = panel("padding:10px 12px;display:flex;flex-direction:column;gap:7px;");
    side.appendChild(slicePanel);
    slicePanel.appendChild(
      el("div", `color:${T.faint};font-size:0.7rem;font-family:${T.sans};text-transform:uppercase;letter-spacing:0.08em;`, "|E| slice position")
    );
    const mkSlider = (label: string) => {
      const wrap = el("div", "display:flex;align-items:center;gap:8px;");
      wrap.appendChild(el("span", `color:${T.faint};font-size:0.72rem;font-family:${T.mono};width:14px;`, label));
      const sl = el("input") as HTMLInputElement;
      sl.type = "range";
      sl.style.cssText = `flex:1;accent-color:${T.accent};height:4px;`;
      wrap.appendChild(sl);
      slicePanel.appendChild(wrap);
      return sl;
    };
    const sliderX = mkSlider("x");
    const sliderZ = mkSlider("z");

    // stats + colorbar
    const hud = panel("padding:12px;display:flex;gap:12px;");
    side.appendChild(hud);
    const bar = el("canvas", "width:60px;height:150px;flex-shrink:0;") as HTMLCanvasElement;
    hud.appendChild(bar);
    const stats = el("div", "flex:1;display:flex;flex-direction:column;justify-content:center;gap:2px;");
    hud.appendChild(stats);
    const sFreq = statRow("frequency");
    const sPeak = statRow("peak");
    const sPatch = statRow("patch");
    const sGrid = statRow("field grid");
    for (const s of [sFreq, sPeak, sPatch, sGrid]) stats.appendChild(s.row);

    const hint = el(
      "div",
      `color:${T.faint};font-size:0.72rem;line-height:1.7;font-family:${T.sans};padding:0 2px;`
    );
    hint.innerHTML =
      "Drag to orbit, scroll to zoom.<br>The glowing planes are |E| of the<br>3D FDTD run (log scale over 3<br>decades): the standing wave under<br>the patch feeds the fringing fields<br>at the radiating edges, which add<br>up to the far-field lobe above.";
    side.appendChild(hint);

    // lighting
    this.scene.add(new THREE.HemisphereLight(0xbcd0ff, 0x281c10, 0.8));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(2.5, 3.2, 1.6);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 0.55);
    rim.position.set(-2.4, -0.8, -2.2);
    this.scene.add(rim);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.8;
    this.controls.minDistance = 0.7;
    this.controls.maxDistance = 8;
    this.controls.target.set(0, 0.25, 0);
    this.controls.addEventListener("start", () => {
      if (this.controls) this.controls.autoRotate = false;
    });

    this.ro = new ResizeObserver(() => {
      const w = viewPanel.clientWidth;
      const h = viewPanel.clientHeight;
      if (w < 2 || h < 2) return;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });
    this.ro.observe(viewPanel);

    // --- data ----------------------------------------------------------------
    const [ffRes, nfRes] = await Promise.allSettled([
      fetch("./data/farfield3d.json").then((r) => (r.ok ? r.json() : Promise.reject())),
      fetch("./data/nearfield3d.json").then((r) => (r.ok ? r.json() : Promise.reject())),
    ]);
    if (ffRes.status === "fulfilled") this.ff = ffRes.value as FarField3DData;
    if (nfRes.status === "fulfilled") this.nf = nfRes.value as NearField3DData;

    if (!this.ff && !this.nf) {
      overlayMessage(
        viewPanel,
        `No data yet —<br><code style="color:${T.copper}">python scripts/export_viz.py</code>`
      );
    }

    // --- build the scene -------------------------------------------------------
    if (this.nf) {
      const g = this.nf.geometry;
      const bw = g.board_mm[1] - g.board_mm[0];
      const bl = g.board_mm[3] - g.board_mm[2];
      this.s = 1 / Math.max(bw, bl);
      this.center.set(
        (g.board_mm[0] + g.board_mm[1]) / 2,
        (g.board_mm[2] + g.board_mm[3]) / 2,
        g.z_gnd_mm
      );
      this.buildAntenna(this.nf);
      this.buildSlices(this.nf, sliderX, sliderZ);
      sPatch.value.textContent = `${(g.patch_mm[1] - g.patch_mm[0]).toFixed(1)}×${(g.patch_mm[3] - g.patch_mm[2]).toFixed(1)} mm`;
      sGrid.value.textContent = this.nf.shape.join("×");
      sFreq.value.textContent = `${(this.nf.freq_hz / 1e9).toFixed(2)} GHz`;
    }
    if (this.ff) {
      this.buildLobe(this.ff, this.useDb);
      const dMax = Math.max(...this.ff.directivity.flat());
      sPeak.value.textContent = `${(10 * Math.log10(dMax)).toFixed(1)} dBi`;
      sFreq.value.textContent = `${(this.ff.freq_hz / 1e9).toFixed(2)} GHz`;
    }
    this.buildStage();

    drawColorbar(bar, VIRIDIS, [
      { t: 0, label: `−${DB_DECADES}0` },
      { t: 0.5, label: `−${(DB_DECADES * 10) / 2}` },
      { t: 1, label: "0 dB" },
    ], "|E|");

    // layer toggles
    mkCheck("far-field lobe", true, (v) => { if (this.lobe) this.lobe.visible = v; });
    mkCheck("|E| slice (vertical)", true, (v) => { if (this.sliceX) this.sliceX.visible = v; });
    mkCheck("|E| slice (horizontal)", true, (v) => { if (this.sliceZ) this.sliceZ.visible = v; });

    const setScale = (db: boolean) => {
      this.useDb = db;
      styleButton(linBtn, { active: !db });
      styleButton(dbBtn, { active: db });
      if (this.ff) this.buildLobe(this.ff, db);
    };
    linBtn.addEventListener("click", () => setScale(false));
    dbBtn.addEventListener("click", () => setScale(true));

    const animate = () => {
      this.rafId = requestAnimationFrame(animate);
      this.controls?.update();
      this.renderer.render(this.scene, this.camera);
    };
    animate();
  }

  /** mm position -> scene coordinates (x->x, y->z, height->y). */
  private toScene(xMm: number, yMm: number, zMm: number): THREE.Vector3 {
    return new THREE.Vector3(
      (xMm - this.center.x) * this.s,
      (zMm - this.center.z) * this.s,
      (yMm - this.center.y) * this.s
    );
  }

  /** The physical antenna: ground, substrate, patch, probe pin. */
  private buildAntenna(nf: NearField3DData): void {
    const g = nf.geometry;
    const grp = new THREE.Group();
    const bw = (g.board_mm[1] - g.board_mm[0]) * this.s;
    const bl = (g.board_mm[3] - g.board_mm[2]) * this.s;
    const hs = g.h_sub_mm * this.s;
    const boardC = this.toScene(
      (g.board_mm[0] + g.board_mm[1]) / 2,
      (g.board_mm[2] + g.board_mm[3]) / 2,
      g.z_gnd_mm
    );

    // ground plane (thin copper sheet)
    const gnd = new THREE.Mesh(
      new THREE.BoxGeometry(bw, 0.004, bl),
      new THREE.MeshStandardMaterial({ color: 0x8a5a28, metalness: 0.85, roughness: 0.4 })
    );
    gnd.position.copy(boardC).add(new THREE.Vector3(0, -0.002, 0));
    grp.add(gnd);

    // substrate (translucent FR-4)
    const sub = new THREE.Mesh(
      new THREE.BoxGeometry(bw, hs, bl),
      new THREE.MeshPhysicalMaterial({
        color: 0x2c4a40,
        transparent: true,
        opacity: 0.45,
        roughness: 0.7,
        transmission: 0.15,
        depthWrite: false,
      })
    );
    sub.position.copy(boardC).add(new THREE.Vector3(0, hs / 2, 0));
    grp.add(sub);

    // patch (copper, slightly above the substrate top)
    const pw = (g.patch_mm[1] - g.patch_mm[0]) * this.s;
    const pl = (g.patch_mm[3] - g.patch_mm[2]) * this.s;
    const patchC = this.toScene(
      (g.patch_mm[0] + g.patch_mm[1]) / 2,
      (g.patch_mm[2] + g.patch_mm[3]) / 2,
      g.z_patch_mm
    );
    const patch = new THREE.Mesh(
      new THREE.BoxGeometry(pw, 0.004, pl),
      new THREE.MeshStandardMaterial({ color: 0xd07f33, metalness: 0.85, roughness: 0.32 })
    );
    patch.position.copy(patchC);
    grp.add(patch);

    // probe feed pin
    const pin = new THREE.Mesh(
      new THREE.CylinderGeometry(0.0045, 0.0045, hs, 12),
      new THREE.MeshStandardMaterial({ color: 0xffd9a0, metalness: 0.9, roughness: 0.25 })
    );
    const pinPos = this.toScene(g.feed_mm[0], g.feed_mm[1], g.z_gnd_mm);
    pin.position.copy(pinPos).add(new THREE.Vector3(0, hs / 2, 0));
    grp.add(pin);

    this.scene.add(grp);
  }

  /** Two translucent |E| slices with draggable positions. */
  private buildSlices(nf: NearField3DData, sliderX: HTMLInputElement, sliderZ: HTMLInputElement): void {
    const [nx, ny, nz] = nf.shape;
    const [dx, dy, dz] = nf.spacing_mm;
    const m = nf.geometry.npml + 1; // crop the CPML frame
    const ix0 = m, ix1 = nx - m, iy0 = m, iy1 = ny - m, iz0 = m, iz1 = nz - m;
    const wY = iy1 - iy0, wZ = iz1 - iz0, wX = ix1 - ix0;
    const eIdx = (i: number, j: number, k: number) => (i * ny + j) * nz + k;

    const mkTexture = (w: number, h: number) => {
      const t = new THREE.DataTexture(
        new Uint8Array(new ArrayBuffer(w * h * 4)), w, h, THREE.RGBAFormat
      );
      t.colorSpace = THREE.SRGBColorSpace;
      t.magFilter = THREE.LinearFilter;
      t.minFilter = THREE.LinearFilter;
      return t;
    };

    // vertical slice: fixed x index, texture axes (y, z)
    this.sliceXTex = mkTexture(wY, wZ);
    const fillX = (ix: number) => {
      const buf = this.sliceXTex!.image.data as Uint8Array;
      let p = 0;
      for (let k = iz0; k < iz1; k++) {
        for (let j = iy0; j < iy1; j++) {
          const t = fieldT(nf.e_mag[eIdx(ix, j, k)], nf.e_max);
          const [r, g, b] = lutColor(VIRIDIS, t);
          buf[p++] = r; buf[p++] = g; buf[p++] = b;
          buf[p++] = Math.round(235 * Math.pow(t, 1.4));
        }
      }
      this.sliceXTex!.needsUpdate = true;
    };

    // horizontal slice: fixed z index, texture axes (x, y)
    this.sliceZTex = mkTexture(wX, wY);
    const fillZ = (iz: number) => {
      const buf = this.sliceZTex!.image.data as Uint8Array;
      let p = 0;
      for (let j = iy0; j < iy1; j++) {
        for (let i = ix0; i < ix1; i++) {
          const t = fieldT(nf.e_mag[eIdx(i, j, iz)], nf.e_max);
          const [r, g, b] = lutColor(VIRIDIS, t);
          buf[p++] = r; buf[p++] = g; buf[p++] = b;
          buf[p++] = Math.round(235 * Math.pow(t, 1.4));
        }
      }
      this.sliceZTex!.needsUpdate = true;
    };

    const sliceMat = (tex: THREE.DataTexture) =>
      new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        side: THREE.DoubleSide,
        depthWrite: false,
      });

    // geometry sized to the cropped interior, positioned via toScene
    const spanY = wY * dy * this.s;
    const spanZ = wZ * dz * this.s;
    const spanX = wX * dx * this.s;
    const cx = ((ix0 + ix1) / 2) * dx;
    const cy = ((iy0 + iy1) / 2) * dy;
    const cz = ((iz0 + iz1) / 2) * dz;

    this.sliceX = new THREE.Mesh(new THREE.PlaneGeometry(spanY, spanZ), sliceMat(this.sliceXTex));
    this.sliceX.rotation.y = -Math.PI / 2; // plane normal -> +x
    this.scene.add(this.sliceX);

    this.sliceZ = new THREE.Mesh(new THREE.PlaneGeometry(spanX, spanY), sliceMat(this.sliceZTex));
    // +PI/2 keeps texture-v aligned with +y_mm (scene +z); DoubleSide anyway
    this.sliceZ.rotation.x = Math.PI / 2;
    this.scene.add(this.sliceZ);

    // default positions: vertical through the feed, horizontal mid-substrate
    const g = nf.geometry;
    const feedI = Math.round(g.feed_mm[0] / dx);
    const midSubK = Math.round((g.z_gnd_mm + g.h_sub_mm / 2) / dz);

    const placeX = (ix: number) => {
      const pos = this.toScene(ix * dx, cy, cz);
      this.sliceX!.position.copy(pos);
      fillX(ix);
    };
    const placeZ = (iz: number) => {
      const pos = this.toScene(cx, cy, iz * dz);
      this.sliceZ!.position.copy(pos);
      fillZ(iz);
    };

    sliderX.min = String(ix0);
    sliderX.max = String(ix1 - 1);
    sliderX.value = String(Math.min(ix1 - 1, Math.max(ix0, feedI)));
    sliderZ.min = String(iz0);
    sliderZ.max = String(iz1 - 1);
    sliderZ.value = String(Math.min(iz1 - 1, Math.max(iz0, midSubK)));
    sliderX.addEventListener("input", () => placeX(parseInt(sliderX.value, 10)));
    sliderZ.addEventListener("input", () => placeZ(parseInt(sliderZ.value, 10)));
    placeX(parseInt(sliderX.value, 10));
    placeZ(parseInt(sliderZ.value, 10));
  }

  /** Far-field lobe floating above the antenna (broadside up). */
  private buildLobe(data: FarField3DData, useDb: boolean): void {
    if (this.lobe) {
      this.scene.remove(this.lobe);
      this.lobe.traverse((o) => {
        const mm = o as THREE.Mesh;
        if (mm.geometry) mm.geometry.dispose();
        if (mm.material) (mm.material as THREE.Material).dispose();
      });
    }
    this.lobe = new THREE.Group();
    const R = 0.62; // lobe radius in scene units

    const nTheta = data.thetas_rad.length;
    const nPhi = data.phis_rad.length;
    const dMax = Math.max(...data.directivity.flat()) || 1;
    const norm = (d: number): number => {
      const lin = d / dMax;
      if (!useDb) return lin;
      const db = 10 * Math.log10(Math.max(lin, 10 ** (this.DB_FLOOR / 10)));
      return Math.max(0, (db - this.DB_FLOOR) / -this.DB_FLOOR);
    };

    const vertices: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];
    for (let ti = 0; ti < nTheta; ti++) {
      for (let pi = 0; pi < nPhi; pi++) {
        const t = norm(data.directivity[ti][pi]);
        const th = data.thetas_rad[ti];
        const ph = data.phis_rad[pi];
        vertices.push(
          R * t * Math.sin(th) * Math.cos(ph),
          R * t * Math.cos(th),
          R * t * Math.sin(th) * Math.sin(ph)
        );
        const [r, g, b] = lutColor(VIRIDIS, t);
        colors.push((r / 255) ** 0.9, (g / 255) ** 0.9, (b / 255) ** 0.9);
      }
    }
    for (let ti = 0; ti < nTheta - 1; ti++) {
      for (let pi = 0; pi < nPhi; pi++) {
        const pj = (pi + 1) % nPhi;
        const a = ti * nPhi + pi;
        const b = ti * nPhi + pj;
        const cc = (ti + 1) * nPhi + pi;
        const dd = (ti + 1) * nPhi + pj;
        indices.push(a, b, cc, b, dd, cc);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    const mesh = new THREE.Mesh(
      geo,
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        roughness: 0.45,
        metalness: 0.08,
        transparent: true,
        opacity: 0.8,
        depthWrite: false,
      })
    );
    this.lobe.add(mesh);
    this.lobe.add(
      new THREE.LineSegments(
        new THREE.WireframeGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.045 })
      )
    );
    // sits on the patch center
    this.lobe.position.set(0, 0.01, 0);
    this.scene.add(this.lobe);
  }

  /** Minimal stage: horizon ring + axis labels. */
  private buildStage(): void {
    const stage = new THREE.Group();
    const ringPts: THREE.Vector3[] = [];
    for (let k = 0; k <= 96; k++) {
      const a = (k / 96) * Math.PI * 2;
      ringPts.push(new THREE.Vector3(0.66 * Math.cos(a), 0, 0.66 * Math.sin(a)));
    }
    stage.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(ringPts),
        new THREE.LineBasicMaterial({ color: 0x303a52, transparent: true, opacity: 0.4 })
      )
    );
    const up = textSprite("z  broadside", "#aebbd6");
    up.position.set(0, 0.78, 0);
    stage.add(up);
    this.scene.add(stage);
  }

  dispose(): void {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    if (this.ro) this.ro.disconnect();
    this.controls?.dispose();
    this.sliceXTex?.dispose();
    this.sliceZTex?.dispose();
    this.renderer.dispose();
  }
}
