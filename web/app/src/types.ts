// Data types for the JSON formats produced by scripts/export_viz.py.

export interface OptimizationData {
  kind: string;
  nx: number;
  ny: number;
  extent_mm: [number, number];
  objective_label: string;
  objective: number[];
  frames: number[][]; // each frame: nx*ny floats, row-major idx = i*ny + j
}

export interface FarField3DData {
  kind: string;
  freq_hz: number;
  thetas_rad: number[];
  phis_rad: number[];
  directivity: number[][]; // [n_theta][n_phi]
  peak: { theta: number; phi: number; d: number };
}

export interface S11Data {
  kind: string;
  freq_hz: number[];
  s11_db_gradenna: number[];
  s11_db_openems?: number[];
  label: string;
}

// WASM module interface (frozen API of web/wasm-kernel).
export interface Fdtd2DInstance {
  set_sigma(sigma: Float32Array): void;
  clear_sigma(): void;
  step(src_i: number, src_j: number, src_val: number): void;
  ez_ptr(): number;
  nx(): number;
  ny(): number;
  dt_seconds(): number;
  reset(): void;
}

export interface GradennaWasmModule {
  default: () => Promise<{ memory?: WebAssembly.Memory }>;
  Fdtd2D: new (
    nx: number,
    ny: number,
    dx_m: number,
    npml: number
  ) => Fdtd2DInstance;
  memory?: WebAssembly.Memory;
}
