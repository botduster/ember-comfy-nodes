#!/usr/bin/env node
/**
 * Runs js/lib/resolution_mp_preview.js over every case from reference.py and requires
 * zero differences from the Python node. With --mutants it also plants known bugs in a
 * copy of the JS and requires each one to be caught.
 *
 *   python3 reference.py > cases.jsonl && node check.mjs cases.jsonl --mutants
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const libPath = join(here, "..", "..", "js", "lib", "resolution_mp_preview.js");
const source = readFileSync(libPath, "utf8");
const cases = readFileSync(process.argv[2], "utf8").trim().split("\n").map((line) => JSON.parse(line));
const PHASE1 = new Set(["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9"]);

async function load(text) {
    return import(`data:text/javascript;base64,${Buffer.from(text).toString("base64")}`);
}

function run(lib) {
    let mismatches = 0, grid25472 = 0, grid25472Mismatches = 0;
    const first = [];
    for (const c of cases) {
        const got = lib.previewText(c.mp, c.aspect_ratio, c.multiple_of);
        const inGrid = PHASE1.has(c.aspect_ratio) && [8, 16, 32, 64].includes(c.multiple_of)
            && Math.round(c.mp * 100) / 100 === c.mp && Number.isInteger(Math.round(c.mp * 100));
        if (inGrid) grid25472++;
        if (got !== c.expected) {
            mismatches++;
            if (inGrid) grid25472Mismatches++;
            if (first.length < 3) first.push({ ...c, got });
        }
    }
    return { cases: cases.length, mismatches, grid25472, grid25472Mismatches, first };
}

const MUTANTS = [
    ["ties-round-up (Math.round)", "return floor % 2 === 0 ? floor : floor + 1;", "return floor + 1;"],
    ["toFixed-for-megapixels", "${pyFixed(actual, 3)} MP", "${actual.toFixed(3)} MP"],
    ["toFixed-for-ratio", "${pyFixed(w / h, 4)}`", "${(w / h).toFixed(4)}`"],
    ["score-tie-last-wins", "if (best === null || score < best[0])", "if (best === null || score <= best[0])"],
    ["ratio-weight-0.3", "const RATIO_WEIGHT = 0.25;", "const RATIO_WEIGHT = 0.3;"],
    ["search-radius-2", "const SEARCH_RADIUS = 3;", "const SEARCH_RADIUS = 2;"],
];

const real = run(await load(source));
console.log(JSON.stringify({ implementation: "real", ...real }));
let ok = real.mismatches === 0 && real.cases > 0;

if (process.argv.includes("--mutants")) {
    for (const [id, anchor, replacement] of MUTANTS) {
        const hits = source.split(anchor).length - 1;
        if (hits !== 1) throw new Error(`mutant ${id}: anchor found ${hits} times`);
        const result = run(await load(source.replace(anchor, replacement)));
        const caught = result.mismatches > 0;
        console.log(JSON.stringify({ mutant: id, caught, mismatches: result.mismatches, example: result.first[0] }));
        ok &&= caught;
    }
}
console.log(ok ? "ALL GOOD" : "FAILURES PRESENT");
process.exit(ok ? 0 : 1);
