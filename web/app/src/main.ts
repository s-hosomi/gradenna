import { OptimizationView } from "./views/OptimizationView";
import { LiveFdtdView } from "./views/LiveFdtdView";
import { Antenna3DView } from "./views/Antenna3DView";
import { S11View } from "./views/S11View";
import { T, el } from "./ui";

const TABS = [
  { id: "optimization", label: "Optimization" },
  { id: "live-fdtd", label: "Live FDTD" },
  { id: "antenna3d", label: "Antenna 3D" },
  { id: "s11", label: "S11" },
] as const;

type TabId = (typeof TABS)[number]["id"];

interface ViewInstance {
  dispose: () => void;
}

// Small inline antenna mark (radiating monopole) for the header.
const LOGO_SVG = `
<svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
  <path d="M12 21 V9" stroke="${T.copper}" stroke-width="2" stroke-linecap="round"/>
  <circle cx="12" cy="7.4" r="2" fill="${T.copper}"/>
  <path d="M7.5 7.5 a6.4 6.4 0 0 1 9 0" stroke="${T.accent}" stroke-width="1.6" stroke-linecap="round" opacity="0.9"/>
  <path d="M5 5 a9.9 9.9 0 0 1 14 0" stroke="${T.accent}" stroke-width="1.6" stroke-linecap="round" opacity="0.5"/>
</svg>`;

class App {
  private activeView: ViewInstance | null = null;
  private contentEl!: HTMLElement;
  private tabBtns = new Map<TabId, HTMLButtonElement>();

  init(): void {
    const root = document.getElementById("app")!;
    root.innerHTML = "";
    root.style.cssText =
      `display:flex;flex-direction:column;height:100vh;background:${T.bg};color:${T.text};` +
      `font-family:${T.sans};overflow:hidden;`;
    document.body.style.margin = "0";
    document.body.style.background = T.bg;

    // header
    const header = el(
      "header",
      `display:flex;align-items:center;padding:0 20px;height:52px;flex-shrink:0;gap:24px;` +
        `background:linear-gradient(180deg, #10141f 0%, #0c0f17 100%);` +
        `border-bottom:1px solid ${T.panelEdge};`
    );
    root.appendChild(header);

    const logo = el("div", "display:flex;align-items:center;gap:10px;");
    logo.innerHTML =
      LOGO_SVG +
      `<span style="font-size:1.05rem;color:${T.text};font-weight:700;letter-spacing:0.01em;">grad<span style="color:${T.copper}">enna</span></span>` +
      `<span style="color:${T.faint};font-size:0.72rem;padding-top:2px;font-family:${T.mono};">differentiable FDTD antenna inverse design</span>`;
    header.appendChild(logo);

    const nav = el("nav", "display:flex;gap:4px;margin-left:auto;height:100%;align-items:stretch;");
    header.appendChild(nav);

    TABS.forEach(({ id, label }) => {
      const btn = el("button") as HTMLButtonElement;
      btn.textContent = label;
      btn.style.cssText = this.tabStyle(false);
      btn.addEventListener("click", () => this.switchTab(id));
      btn.addEventListener("mouseenter", () => {
        if (!btn.dataset.active) btn.style.color = T.text;
      });
      btn.addEventListener("mouseleave", () => {
        if (!btn.dataset.active) btn.style.color = T.dim;
      });
      nav.appendChild(btn);
      this.tabBtns.set(id, btn);
    });

    this.contentEl = el("main", "flex:1;min-height:0;overflow:hidden;");
    root.appendChild(this.contentEl);

    const footer = el(
      "footer",
      `height:26px;border-top:1px solid ${T.panelEdge};display:flex;align-items:center;` +
        `padding:0 20px;gap:16px;flex-shrink:0;background:#0c0f17;`
    );
    footer.innerHTML =
      `<span style="color:${T.faint};font-size:0.7rem;font-family:${T.mono};">gradenna · MIT</span>` +
      `<a href="https://github.com/s-hosomi/gradenna" target="_blank" rel="noopener" ` +
      `style="color:${T.faint};font-size:0.7rem;text-decoration:none;font-family:${T.mono};">GitHub ↗</a>` +
      `<span style="margin-left:auto;color:${T.faint};font-size:0.7rem;font-family:${T.mono};">` +
      `2D kernel: Rust → wasm · data: JAX FDTD</span>`;
    root.appendChild(footer);

    this.switchTab("optimization");
  }

  private tabStyle(active: boolean): string {
    return [
      "background:transparent",
      `border-bottom:2px solid ${active ? T.copper : "transparent"}`,
      "border-top:none",
      "border-left:none",
      "border-right:none",
      `color:${active ? T.text : T.dim}`,
      "padding:0 14px",
      "cursor:pointer",
      "font-size:0.8rem",
      `font-family:${T.sans}`,
      `font-weight:${active ? 600 : 400}`,
      "height:100%",
      "transition:color .15s,border-color .15s",
    ].join(";");
  }

  private async switchTab(id: TabId): Promise<void> {
    if (this.activeView) {
      this.activeView.dispose();
      this.activeView = null;
    }
    this.tabBtns.forEach((btn, tabId) => {
      const active = tabId === id;
      btn.style.cssText = this.tabStyle(active);
      if (active) btn.dataset.active = "1";
      else delete btn.dataset.active;
    });

    this.contentEl.innerHTML = "";
    const viewContainer = el("div", "width:100%;height:100%;opacity:0;transition:opacity .18s ease;");
    this.contentEl.appendChild(viewContainer);
    requestAnimationFrame(() => (viewContainer.style.opacity = "1"));

    let view: ViewInstance;
    switch (id) {
      case "optimization": {
        const v = new OptimizationView(viewContainer);
        await v.init();
        view = v;
        break;
      }
      case "live-fdtd": {
        const v = new LiveFdtdView(viewContainer);
        await v.init();
        view = v;
        break;
      }
      case "antenna3d": {
        const v = new Antenna3DView(viewContainer);
        await v.init();
        view = v;
        break;
      }
      case "s11": {
        const v = new S11View(viewContainer);
        await v.init();
        view = v;
        break;
      }
    }
    this.activeView = view!;
  }
}

new App().init();
