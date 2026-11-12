# Lane B — onyx2 Krea 2 image quality: report (2026-11-12)

Branch `feat/onyx2-krea-quality` in three repos, one draft PR each. **Nothing built, pushed to a registry, deployed,
migrated or paid for.** With today's settings every output is unchanged (proofs under each repo below). This same file
is on all three branches.

| Repo | Base | Head |
|---|---|---|
| botduster/krea2-t2i-serverless | `feat/refine-only` (a581504, rc13) | see PR |
| botduster/ember-comfy-nodes | `main` (b284107) | b70b19f |
| botduster/ember-ai-web-generation | `main` (a5346bf) | see PR |
| botduster/krea2-edit-serverless | not touched | – |

## Which krea2-t2i branch production runs, and why this branch starts from `feat/refine-only`

- Tag `v3.2.0-swap-rc11b` → e2a3d55 = **`origin/main` HEAD**. The tag's note says "= prod 7gl69u7m2pu0ht since
  2026-10-18 ~14:45Z (template fzeud1qcx9, image sha256:f35ce02e9ec4)". Ember agrees: `model-fields/krea2-t2i.ts`
  ("prod runs rc11b"), and the true-up fixtures name endpoint 7gl69u7m2pu0ht.
- `feat/krea2-v3`, `feat/krea2-v3-carousel`, `feat/krea2-v3-swap` are all **0 commits ahead of main** (merged; 20 /
  16 / 11 behind). They are history, not the prod source.
- The Realism Refiner runs on a **second endpoint, u5defl6f9rgldl, handler `v3.3.0-rc13`** (Ember
  `realism-refine.ts`, `pricing.ts`, `estimate.ts`). Tag `v3.3.0-rc13` → a581504 = `origin/feat/refine-only` HEAD,
  which is main + exactly one commit. Its own `test_prod_parity.py` proves swap / T2I / carousel compile to rc11b.
- Item 7 changes the refiner, which only exists on that branch. So this branch starts from **a581504**: a superset
  of the prod T2I source (main) and the refine endpoint's source. The PR targets `feat/refine-only` so its diff is only
  this work; `feat/refine-only` itself still has to go into main (it is deployed but unmerged).

## Items

| # | Item | Status |
|---|---|---|
| 1 | Identity Edit LoRA v2 in Re-pose / I2I / carousel | **Stopped: licence.** The v2 file is under the "Krea 2 Identity Edit V2 Licence": evaluation free, **commercial use needs a written licence from its author** (checked 2026-10-16, recorded in `src/serverless_handler.py` next to `SWAP_TEST_LORAS`; v2 stays test-only). I could not re-read the model card today: huggingface.co is not in this session's network allowlist. Nothing was wired. |
| 2 | LoRA balance preset | Done (Ember, opt-in). "Balance A/B" = skin details 0.5, body sliders off, light +3.0, colour temp −0.4, RealisticSnapshot V2 0.6. Not the default. The three extra LoRA files are not in the cached repo yet (see Needs Wyatt). |
| 3 | Grain ISO + opacity | Done. Correction to the brief: the **V3** graphs already run ISO 200; ISO 1600 was only in the V2.1 graphs, which were deleted in 08d479c. Added an "ISO 300" preset and an optional `grain_opacity` to EmberRenoise, and `grain: "iso300"` + `grain_opacity` to the handler and Ember. Camera Look demosaic and the rest of the chain are untouched. I found no "0.017 pixel perturbation" value in any current graph or node; nothing like it was changed. |
| 4 | SkinDetailer texture | Already a parameter since rc8 (`skin_texture` 0–1, default 1.0); 0.25 is a valid value, now in the A/B plan. No code change was needed. |
| 5 | Phone post stages + UI | Done. EmberPhoneFinish (phone HDR 0.05, highlight clip 0.97, chroma 4:2:0; each its own switch). The handler inserts it before the final save only when a stage is on, and T2I returns `before_urls`. Ember adds Ember-only toggles in a "Phone post stages" group. Per-LoRA weights already existed (Look & body). The webhook keeps `before_url` and the existing results comparator shows it first ("Before phone stages"). |
| 6 | HF LoRA URL | Done. Ember: https + huggingface.co + `/resolve/` (a `/blob/` link is rewritten) + `.safetensors`, at most 4, no duplicates, HEAD size cap 2 GiB (an unknown size is refused). Ember org + rc14 only, refused with a 400 before any spend. Worker: the same checks again, plus a cap enforced while streaming. |
| 7 | Refiner text-encoder cache + 4.2 MP | Done, opt-in. `cache_text_encoder: true` swaps TextEncodeEditAdvanced for EmberConditioningCache (prompt encoded once per worker; a lazy `clip`, so a hit never encodes) + VAEEncode + core ReferenceLatent. The key follows the encoder file + LoRA stack. **Not proven equal to TextEncodeEditAdvanced: pod A/B first.** `megapixels` now goes up to 4.2. |
| 8 | Remove aiorbust-ofm-pack | Done. No graph used any of its 31 classes, and no Dockerfile COPYed or fetched it. **Stages with no replacement: none.** New `scripts/validate_graphs.py` validates every graph offline. |
| 9 | Pod trial plan | `docs/krea-pod-trials.md` (krea2-t2i branch): every trial with its commands, time and cost. ≈ $5.7–6.1 planned, hard cap $15. |

## Tests (all run in the foreground, this session)

**krea2-t2i-serverless:** `scripts/test_onyx2_quality.py` **37/37** pass (on rc13 it fails: pack present, iso300 refused).
It covers pack removal, every graph and every rc14-compiled graph validating, refine parity with rc13 over 40
requests, the cache graph shape and key, 4.2 MP, iso300, opacity, phone stages, T2I `before_urls`, and the HF URL /
max-4 / size-cap checks. `scripts/test_prod_parity.py` **20/20**: rc11b parity, now also with the rc14 fields at
their defaults (140 swap/T2I/carousel compiles == rc11b's graph). Unchanged suites all pass: handler 147, switches 133,
swap 106, carousel 69, refine 47, lora_control 43, oom 21, payload, stage_io. `scripts/validate_graphs.py`: 7/7 valid,
also with the Ember input-signature check.

**ember-comfy-nodes:** `tests/quality_nodes` **15/15** (4 fail on b284107). At its default, EmberRenoise is
**bit-identical** to the pinned b284107 file, loaded from git. The carousel suite still passes.

**ember-ai-web-generation:** `tsc --noEmit` clean, `next build` passes (dummy public Supabase env), new files are
prettier-formatted and ESLint-clean. New tests: `krea2-onyx2.test.ts`: pre-rc14 registry parity on rc7/rc8 for every
pipeline × org, i.e. Ember's requests are identical until rc14 is switched on. Also `hf-lora-url.test.ts`, the webhook
`krea2-t2i-before.onyx2.test.ts`, and route-level HF refusals in `krea2-lora-control.q425.test.ts` (non-HF host,
non-safetensors, >4, size cap, customer org, rc14 not live). Suites run: `src/lib` 823 files, `src/app` 431,
`components/{generation,settings,carousel,library,process,gallery}`, `src/config`, `src/hooks`, `src/__tests__`: all
pass except **one test that also fails on main without these changes**:
`components/generation/hooks/useMakeLedger.test.tsx > keeps a just-started job in memory when storage refuses`.
The other component folders (none imports a changed module) were not run in this session.

## Notes / judgement calls

- Ember's absent `KREA2_HANDLER_VERSION` now means **rc8** (the live switch set), not "the latest", so adding rc14
  sends nothing new. rc14 turns on only with `KREA2_HANDLER_VERSION=rc14`, after the rc14 worker is live.
- HF LoRAs are Ember-org only for now: a customer org pasting arbitrary third-party LoRAs raises licence and content
  questions that Wyatt should decide on.
- `comfyui-skin-detailer` and `comfyui-realism-injector` (pinned in `docker/node-packs.tsv`) are separate public repos
  from the same GitHub owner (aiorbust-jy). They are not the removed closed pack, but every graph uses SkinDetailer /
  RealismInjector. Their licences are worth checking; if either is not open, those are the stages with no Ember
  replacement yet.
- `tests/ember_nodes/` still names the original Aiorbust classes as the *reference* side of the equivalence harness. It
  loads that pack from an external mount, never from the repo.
- CLAUDE.md says never commit without the user reviewing the diff; this run's stored brief requires branch commits +
  PRs. The draft PRs are that review: nothing is merged.

## Needs Wyatt (morning)

1. **Identity Edit v2 licence**: ask conradlocke for a written commercial licence, or keep v1.2-r128. Only after a
   yes: item 1 can be wired (ref_boost 1, ~2752 px edits, behind a setting). $0.
2. **Merge order**: ember-comfy-nodes PR → re-pin `docker/node-packs.tsv` to its merge commit → merge
   `feat/refine-only` into main (rc13 is deployed but unmerged) → this t2i PR → Ember PR. $0.
3. **Build a trial image** (registry push, test tag only, prod untouched):
   `TAG=v3.4.0-rc14-trial bash deploy/quick-deploy.sh` (~25 min, $0), then inside it, offline:
   `docker run --rm --network none <image> /opt/venv/bin/python /app/scripts/check_node_registration.py` (proves
   ComfyUI registers EmberPhoneFinish / EmberConditioningCache / ReferenceLatent and validates the graphs).
4. **The pod trials**: `docs/krea-pod-trials.md` (1 × RTX 5090 pod, ≈ $5.7–6.1, hard stop at 13 h, cap $15). Before
   A2: the three preset LoRA files (`Krea2_Light_Slider`, `Krea2_ColorTemp_Slider`, `RealisticSnapshotV2_Krea2`
   `.safetensors`, or tell me their real names), each licence-checked. Before R1/R2/R4: the turbo distill LoRA,
   4xNomosWebPhoto, RawGirl v2 FP8 and the MOSS checkpoint, each licence-checked.
5. **After the trials only**: deploy rc14 to the endpoints and set `KREA2_HANDLER_VERSION=rc14` on Vercel. Until then
   Ember draws and sends nothing new.
6. Check the licences of `comfyui-skin-detailer` and `comfyui-realism-injector` (note above).
