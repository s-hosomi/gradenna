import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { FarField3DData } from "../types";
import { VIRIDIS, lutColor, drawColorbar } from "../colormaps";
import { T, el, panel, sectionTitle, button, styleButton, statRow, overlayMessage } from "../ui";

// Physics: theta measured from +z (broadside), phi from +x.
// Scene: three.js is Y-up, so broadside (+z physics) maps to +Y.
function dirVector(theta: number, phi: number, r: number): THREE.Vector3 {
  return new THREE.Vector3(
    r * Math.sin(theta) * Math.cos(phi),
    r * Math.cos(theta),
    r * Math.sin(theta) * Math.sin(phi)
  );
}

function textSprite(text: string, color = "#9aa3b5", px = 26): THREE.Sprite {
  const pad = 8;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  ctx.font = `${px}px ${T.sans}`;
  const w = Math.ceil(ctx.measureText(text).width) + pad * 2;
  const h = px + pad * 2;
  canvas.width = w * 2;
  canvas.height = h * 2;
  const c2 = canvas.getContext("2d")!;
  c2.scale(2, 2);
  c2.font = `${px}px ${T.sans}`;
  c2.fillStyle = color;
  c2.textBaseline = "middle";
  c2.fillText(text, pad, h / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = 4;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(mat);
  const scale = 0.0035 * px;
  sprite.scale.set((w / h) * scale, scale, 1);
  return sprite;
}

export class FarField3DView {
  private container: HTMLElement;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  private renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  private controls: OrbitControls | null = null;
  private lobe: THREE.Group | null = null;
  private rafId: number | null = null;
  private ro: ResizeObserver | null = null;
  private data: FarField3DData | null = null;
  private useDb = false;
  private readonly DB_FLOOR = -30;

  constructor(container: HTMLElement) {
    this.container = container;
    this.camera.position.set(1.9, 1.35, 1.9);
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
    head.appendChild(sectionTitle("Far-field pattern", "NTFF directivity of the 2.45 GHz patch"));
    const spacer = el("div", "flex:1;");
    head.appendChild(spacer);
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

    const side = el("div", "width:230px;flex-shrink:0;display:flex;flex-direction:column;gap:10px;");
    row.appendChild(side);
    const hud = panel("padding:12px;display:flex;gap:12px;");
    side.appendChild(hud);
    const bar = el("canvas", "width:60px;height:170px;flex-shrink:0;") as HTMLCanvasElement;
    hud.appendChild(bar);
    const stats = el("div", "flex:1;display:flex;flex-direction:column;justify-content:center;gap:2px;");
    hud.appendChild(stats);
    const sFreq = statRow("frequency");
    const sPeak = statRow("peak D₀");
    const sPeakDb = statRow("peak");
    const sDir = statRow("direction");
    for (const s of [sFreq, sPeak, sPeakDb, sDir]) stats.appendChild(s.row);

    const hint = el(
      "div",
      `color:${T.faint};font-size:0.72rem;line-height:1.7;font-family:${T.sans};padding:0 2px;`
    );
    hint.innerHTML =
      "Drag to orbit, scroll to zoom.<br>The lobe radius is the directivity<br>D(θ,φ); broadside (+z) points up.<br>The patch sits in the dark disc.";
    side.appendChild(hint);

    // --- lighting -------------------------------------------------------------
    this.scene.add(new THREE.HemisphereLight(0xbcd0ff, 0x281c10, 0.85));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(2.5, 3.2, 1.6);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 0.6);
    rim.position.set(-2.4, -0.8, -2.2);
    this.scene.add(rim);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 1.0;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 8;
    this.controls.addEventListener("start", () => {
      if (this.controls) this.controls.autoRotate = false;
    });

    this.buildStage();

    this.ro = new ResizeObserver(() => {
      const w = viewPanel.clientWidth;
      const h = viewPanel.clientHeight;
      if (w < 2 || h < 2) return;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });
    this.ro.observe(viewPanel);

    // --- data -------------------------------------------------------------------
    try {
      const resp = await fetch("./data/farfield3d.json");
      if (!resp.ok) throw new Error(String(resp.status));
      this.data = (await resp.json()) as FarField3DData;
    } catch {
      overlayMessage(
        viewPanel,
        `No data yet —<br><code style="color:${T.copper}">python scripts/export_viz.py</code>`
      );
    }

    const refreshBar = () => {
      if (!this.data) return;
      const dMax = Math.max(...this.data.directivity.flat());
      const peakDbi = 10 * Math.log10(dMax);
      if (this.useDb) {
        const f = this.DB_FLOOR;
        drawColorbar(bar, VIRIDIS, [
          { t: 0, label: `${(peakDbi + f).toFixed(0)}` },
          { t: 0.5, label: `${(peakDbi + f / 2).toFixed(0)}` },
          { t: 1, label: `${peakDbi.toFixed(1)}` },
        ], "dBi");
      } else {
        drawColorbar(bar, VIRIDIS, [
          { t: 0, label: "0" },
          { t: 0.5, label: (dMax / 2).toFixed(1) },
          { t: 1, label: dMax.toFixed(1) },
        ], "D");
      }
    };

    if (this.data) {
      const d = this.data;
      this.buildLobe(d, this.useDb);
      refreshBar();
      const dMax = Math.max(...d.directivity.flat());
      sFreq.value.textContent = `${(d.freq_hz / 1e9).toFixed(2)} GHz`;
      sPeak.value.textContent = d.peak.d.toFixed(2);
      sPeakDb.value.textContent = `${(10 * Math.log10(dMax)).toFixed(1)} dBi`;
      sDir.value.textContent = `θ=${((d.peak.theta * 180) / Math.PI).toFixed(0)}° (broadside)`;
    }

    const setScale = (db: boolean) => {
      this.useDb = db;
      styleButton(linBtn, { active: !db });
      styleButton(dbBtn, { active: db });
      if (this.data) this.buildLobe(this.data, db);
      refreshBar();
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

  /** Static stage: graticule sphere, principal-plane rings, axes, ground disc. */
  private buildStage(): void {
    const stage = new THREE.Group();

    const ringMat = new THREE.LineBasicMaterial({
      color: 0x44506a,
      transparent: true,
      opacity: 0.45,
    });
    const circle = (axis: "xy" | "xz" | "yz", r: number, mat = ringMat) => {
      const pts: THREE.Vector3[] = [];
      for (let k = 0; k <= 96; k++) {
        const a = (k / 96) * Math.PI * 2;
        if (axis === "xz") pts.push(new THREE.Vector3(r * Math.cos(a), 0, r * Math.sin(a)));
        else if (axis === "xy") pts.push(new THREE.Vector3(r * Math.cos(a), r * Math.sin(a), 0));
        else pts.push(new THREE.Vector3(0, r * Math.sin(a), r * Math.cos(a)));
      }
      return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
    };

    // principal great circles at the lobe scale
    stage.add(circle("xz", 1.0), circle("xy", 1.0), circle("yz", 1.0));
    // faint radius reference rings in the horizon plane
    const faint = new THREE.LineBasicMaterial({ color: 0x303a52, transparent: true, opacity: 0.3 });
    stage.add(circle("xz", 0.5, faint), circle("xz", 0.75, faint), circle("xz", 0.25, faint));

    // axes
    const mkAxis = (to: THREE.Vector3, color: number) => {
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), to]);
      return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }));
    };
    stage.add(mkAxis(new THREE.Vector3(1.35, 0, 0), 0x6a7894));
    stage.add(mkAxis(new THREE.Vector3(0, 1.35, 0), 0x8b9cc0));
    stage.add(mkAxis(new THREE.Vector3(0, 0, 1.35), 0x6a7894));

    const lx = textSprite("x  (φ=0°)");
    lx.position.set(1.5, 0, 0);
    const ly = textSprite("z  broadside", "#aebbd6");
    ly.position.set(0, 1.48, 0);
    const lz = textSprite("y  (φ=90°)");
    lz.position.set(0, 0, 1.5);
    stage.add(lx, ly, lz);

    // ground-plane disc hint (the patch substrate)
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(0.55, 48),
      new THREE.MeshStandardMaterial({
        color: 0x141a26,
        roughness: 0.9,
        metalness: 0.1,
        transparent: true,
        opacity: 0.85,
        side: THREE.DoubleSide,
      })
    );
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = -0.012;
    stage.add(disc);
    const patch = new THREE.Mesh(
      new THREE.PlaneGeometry(0.3, 0.23),
      new THREE.MeshStandardMaterial({
        color: 0xb06a28,
        roughness: 0.45,
        metalness: 0.7,
        side: THREE.DoubleSide,
      })
    );
    patch.rotation.x = -Math.PI / 2;
    patch.position.y = -0.006;
    stage.add(patch);

    this.scene.add(stage);
  }

  private buildLobe(data: FarField3DData, useDb: boolean): void {
    if (this.lobe) {
      this.scene.remove(this.lobe);
      this.lobe.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
        if (m.material) (m.material as THREE.Material).dispose();
      });
    }
    this.lobe = new THREE.Group();

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
        const v = dirVector(data.thetas_rad[ti], data.phis_rad[pi], t);
        vertices.push(v.x, v.y, v.z);
        const [r, g, b] = lutColor(VIRIDIS, t);
        // slight gamma lift keeps the low-D regions readable
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

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      roughness: 0.42,
      metalness: 0.08,
    });
    this.lobe.add(new THREE.Mesh(geo, mat));

    // faint wireframe to articulate the surface
    const wire = new THREE.LineSegments(
      new THREE.WireframeGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.05 })
    );
    this.lobe.add(wire);

    // peak direction marker
    const peakDir = dirVector(data.peak.theta, data.peak.phi, 1).normalize();
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      peakDir.clone().multiplyScalar(1.18),
    ]);
    this.lobe.add(
      new THREE.Line(
        lineGeo,
        new THREE.LineDashedMaterial({ color: 0xffd9a0, transparent: true, opacity: 0.7 })
      )
    );
    const tip = new THREE.Mesh(
      new THREE.SphereGeometry(0.022, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffd9a0 })
    );
    tip.position.copy(peakDir.clone().multiplyScalar(1.18));
    this.lobe.add(tip);

    this.scene.add(this.lobe);
  }

  dispose(): void {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    if (this.ro) this.ro.disconnect();
    this.controls?.dispose();
    this.renderer.dispose();
  }
}
