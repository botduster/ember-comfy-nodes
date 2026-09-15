/**
 * Ember Resolution (MP) preview maths: the browser-side twin of
 * ember_nodes/resolution_mp.py.
 *
 * The widget has to show exactly what the node will output. That rules out
 * "close enough" JavaScript:
 *   - Python's round() rounds halves to even; Math.round rounds them up.
 *   - Python's "%.3f" rounds the exact binary value halves to even; toFixed picks the
 *     larger neighbour on a tie.
 * Both are reproduced exactly below. tests/resolution_preview runs this file against
 * the Python node over the whole widget grid and requires zero differences, so any
 * change here must keep that test green.
 *
 * Pure functions, no DOM, no ComfyUI imports, so Node can load it for the test.
 */

export const FROM_IMAGE = "from image (input)";

export const ASPECT_PRESETS = [
    [FROM_IMAGE, null],
    ["1:1", [1, 1]],
    ["4:3", [4, 3]],
    ["3:4", [3, 4]],
    ["3:2", [3, 2]],
    ["2:3", [2, 3]],
    ["16:9", [16, 9]],
    ["9:16", [9, 16]],
    ["21:9", [21, 9]],
    ["5:4", [5, 4]],
    ["4:5", [4, 5]],
];

const MIN_SIDE = 64;
const MAX_SIDE = 8192;
const RATIO_WEIGHT = 0.25;
const SEARCH_RADIUS = 3;

/** Python's round(x) for a non-negative float: nearest integer, ties to even. */
export function roundHalfEven(x) {
    const floor = Math.floor(x);
    const frac = x - floor; // exact for doubles in this range
    if (frac > 0.5) return floor + 1;
    if (frac < 0.5) return floor;
    return floor % 2 === 0 ? floor : floor + 1;
}

function grid(value, multiple) {
    return Math.max(multiple, roundHalfEven(value / multiple) * multiple);
}

/** Same search, same scoring, same tie-break as snap_to_budget in resolution_mp.py. */
export function snapToBudget(megapixels, ar, multiple) {
    const targetPx = Math.max(1.0, megapixels * 1_000_000.0);
    const centreH = grid(Math.sqrt(targetPx / ar), multiple);

    let best = null; // [score, w, h]
    for (let dh = -SEARCH_RADIUS; dh <= SEARCH_RADIUS; dh++) {
        const h = centreH + dh * multiple;
        if (h < MIN_SIDE || h > MAX_SIDE) continue;
        const centreW = grid(h * ar, multiple);
        for (let dw = -SEARCH_RADIUS; dw <= SEARCH_RADIUS; dw++) {
            const w = centreW + dw * multiple;
            if (w < MIN_SIDE || w > MAX_SIDE) continue;
            const mpErr = Math.abs(w * h - targetPx) / targetPx;
            const ratioErr = Math.abs((w / h) - ar) / ar;
            const score = mpErr + RATIO_WEIGHT * ratioErr;
            if (best === null || score < best[0]) best = [score, w, h];
        }
    }

    if (best === null) {
        const h = Math.max(MIN_SIDE, Math.min(MAX_SIDE, grid(Math.sqrt(targetPx / ar), multiple)));
        const w = Math.max(MIN_SIDE, Math.min(MAX_SIDE, grid(h * ar, multiple)));
        return [w, h];
    }
    return [best[1], best[2]];
}

/** Python's format(x, f".{digits}f"): exact binary value, rounded half to even. */
export function pyFixed(x, digits) {
    const view = new DataView(new ArrayBuffer(8));
    view.setFloat64(0, x);
    const hi = view.getUint32(0);
    const lo = view.getUint32(4);
    const negative = hi >>> 31 === 1;
    const biased = (hi >>> 20) & 0x7ff;
    let mantissa = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
    let exponent;
    if (biased === 0) {
        exponent = -1074;
    } else {
        mantissa |= 1n << 52n;
        exponent = biased - 1075;
    }
    // |x| * 10^digits = mantissa * 2^exponent * 10^digits = num / den
    let num = mantissa * 10n ** BigInt(digits);
    let den = 1n;
    if (exponent >= 0) num <<= BigInt(exponent);
    else den <<= BigInt(-exponent);
    let q = num / den;
    const twiceRemainder = 2n * (num % den);
    if (twiceRemainder > den || (twiceRemainder === den && (q & 1n) === 1n)) q += 1n;
    const padded = q.toString().padStart(digits + 1, "0");
    const body = digits > 0 ? `${padded.slice(0, -digits)}.${padded.slice(-digits)}` : padded;
    return (negative ? "-" : "") + body;
}

/** Python's format(x, f"+.{digits}f"). */
export function pySignedFixed(x, digits) {
    const text = pyFixed(x, digits);
    return text.startsWith("-") ? text : `+${text}`;
}

/** The preview line, in the reference node's format. */
export function previewText(megapixels, aspectRatio, multipleOf) {
    const preset = ASPECT_PRESETS.find(([label]) => label === aspectRatio);
    if (!preset || preset[1] === null) {
        return "from image — computed at run time";
    }
    const ar = preset[1][0] / preset[1][1];
    const [w, h] = snapToBudget(Number(megapixels), ar, Math.trunc(Number(multipleOf)));
    const actual = (w * h) / 1_000_000.0;
    const drift = megapixels ? ((actual - megapixels) / megapixels) * 100.0 : 0.0;
    return `${w} x ${h}  ·  ${pyFixed(actual, 3)} MP (${pySignedFixed(drift, 1)}%)  ·  ${pyFixed(w / h, 4)}`;
}
