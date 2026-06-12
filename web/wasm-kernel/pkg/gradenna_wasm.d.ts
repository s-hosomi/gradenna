/* tslint:disable */
/* eslint-disable */

/**
 * Browser-facing 2D TM FDTD solver.
 *
 * The exported API is frozen; see the crate README for usage from JS.
 */
export class Fdtd2D {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Reset the material to vacuum.
     */
    clear_sigma(): void;
    /**
     * Time step in seconds.
     */
    dt_seconds(): number;
    /**
     * Pointer to the row-major `nx*ny` Ez array in wasm linear memory.
     */
    ez_ptr(): number;
    /**
     * Vacuum-initialized solver. `dx_m` is the (square) cell size in metres,
     * `npml` the CPML thickness in cells per side.
     */
    constructor(nx: number, ny: number, dx_m: number, npml: number);
    nx(): number;
    ny(): number;
    /**
     * Zero all fields (keeps the conductivity map).
     */
    reset(): void;
    /**
     * Set the conductivity map (S/m), row-major `i*ny + j`, length `nx*ny`.
     */
    set_sigma(sigma: Float32Array): void;
    /**
     * Advance one time step, soft-sourcing Ez at `(src_i, src_j)`.
     */
    step(src_i: number, src_j: number, src_val: number): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_fdtd2d_free: (a: number, b: number) => void;
    readonly fdtd2d_clear_sigma: (a: number) => void;
    readonly fdtd2d_dt_seconds: (a: number) => number;
    readonly fdtd2d_ez_ptr: (a: number) => number;
    readonly fdtd2d_new: (a: number, b: number, c: number, d: number) => number;
    readonly fdtd2d_nx: (a: number) => number;
    readonly fdtd2d_ny: (a: number) => number;
    readonly fdtd2d_reset: (a: number) => void;
    readonly fdtd2d_set_sigma: (a: number, b: number, c: number) => void;
    readonly fdtd2d_step: (a: number, b: number, c: number, d: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
