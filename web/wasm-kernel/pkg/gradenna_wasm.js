/* @ts-self-types="./gradenna_wasm.d.ts" */

/**
 * Browser-facing 2D TM FDTD solver.
 *
 * The exported API is frozen; see the crate README for usage from JS.
 */
export class Fdtd2D {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Fdtd2DFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_fdtd2d_free(ptr, 0);
    }
    /**
     * Reset the material to vacuum.
     */
    clear_sigma() {
        wasm.fdtd2d_clear_sigma(this.__wbg_ptr);
    }
    /**
     * Time step in seconds.
     * @returns {number}
     */
    dt_seconds() {
        const ret = wasm.fdtd2d_dt_seconds(this.__wbg_ptr);
        return ret;
    }
    /**
     * Pointer to the row-major `nx*ny` Ez array in wasm linear memory.
     * @returns {number}
     */
    ez_ptr() {
        const ret = wasm.fdtd2d_ez_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Vacuum-initialized solver. `dx_m` is the (square) cell size in metres,
     * `npml` the CPML thickness in cells per side.
     * @param {number} nx
     * @param {number} ny
     * @param {number} dx_m
     * @param {number} npml
     */
    constructor(nx, ny, dx_m, npml) {
        const ret = wasm.fdtd2d_new(nx, ny, dx_m, npml);
        this.__wbg_ptr = ret;
        Fdtd2DFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    nx() {
        const ret = wasm.fdtd2d_nx(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    ny() {
        const ret = wasm.fdtd2d_ny(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Zero all fields (keeps the conductivity map).
     */
    reset() {
        wasm.fdtd2d_reset(this.__wbg_ptr);
    }
    /**
     * Set the conductivity map (S/m), row-major `i*ny + j`, length `nx*ny`.
     * @param {Float32Array} sigma
     */
    set_sigma(sigma) {
        const ptr0 = passArrayF32ToWasm0(sigma, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.fdtd2d_set_sigma(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Advance one time step, soft-sourcing Ez at `(src_i, src_j)`.
     * @param {number} src_i
     * @param {number} src_j
     * @param {number} src_val
     */
    step(src_i, src_j, src_val) {
        wasm.fdtd2d_step(this.__wbg_ptr, src_i, src_j, src_val);
    }
}
if (Symbol.dispose) Fdtd2D.prototype[Symbol.dispose] = Fdtd2D.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_bbadd78c1bac3a77: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./gradenna_wasm_bg.js": import0,
    };
}

const Fdtd2DFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_fdtd2d_free(ptr, 1));

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('gradenna_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
