# Continuity mode for FastH3 — PR notes

Adds an optional **continuity mode** to the FastH3 channel as a config-only,
guarded flag on top of the existing hard-cut port. Default off; the hard-cut
path is byte-for-byte unchanged and all 510 existing tests pass untouched.

## What it does

When `inference.continuity: true`, the independent-clip channel becomes one
continuous stream:

- **FL2VA anchor chain** — clip 0 is plain T2VA; every clip after is generated
  anchored on the previous clip's last frame (`InputConfig(pil_image=...)`),
  threaded `submit → job → _generate_clip → _request` alongside prompt/seed.
- **Seam stitch** (`fasth3_seam.py`, pure numpy) — each clip's last `seam_frames`
  are held and crossfaded onto the next clip's head: video in linear light with
  complementary weights (no midpoint flash), audio equal-power overlap-add.
- **Per-clip colour-match locked to clip 0** — continuation clips are shifted by
  one per-channel offset onto clip 0's last-frame mean, so exposure cannot
  ratchet across a long chain.
- **FL2VA warm-up** — `_warmup` warms the FL2VA shape too (grey anchor) when
  continuity is on, so the first continuation clip does not eat a ~20 s compile
  stall live.
- **Shorter steady clip** — continuity uses `continuity_clip_seconds` (default
  5.167 s) instead of the hard-cut `clip_seconds` (14.375 s): the anchor is a
  single still, so re-anchoring more often drifts less and keeps the builder
  further ahead of playout.

Reuses Ruixing's existing machinery unchanged: held-prompt indefinite loop
(`_take_prompt`), double-buffer lookahead, 256-token prompt pad, paced 24 fps
emitter (extended for seams).

## Changed files (all continuity branches guarded by `continuity_enabled`)

- `fasth3.py` — config parse; class-level `continuity_enabled`/`seam_frames`
  defaults; `_snapshot` reports `continuity`; anchor threaded through
  `submit`/`_generate_clip`/`_request`; `_colour_match_clip`; `_stitch_seam` +
  `_emit_paced` seam branch; FL2VA `_warmup` + `_grey_anchor`; `_clip0_reference`
  reset.
- `fasth3_types.py` — `StateUpdate.continuity` field; reworded `ClipStarted` doc.
- `fasth3.yaml` — `inference.continuity`/`continuity_clip_seconds`/`seam_frames`.
- `README.md` — a "Continuity mode" subsection.

## New files

- `fasth3_seam.py` — the seam math (colour-match, linear-light blend, equal-power
  audio), pure numpy, imported lazily so schema render stays torch/numpy-free.
- `tests/test_continuity.py` — 16 CPU-only tests.

## CPU-only validation (no GPU used)

`PYTHONPATH=. python3 -m pytest tests/ -q` → **526 passed, 1 skipped**.

- FL2VA anchor: passed for index > 0, `None` for clip 0 (chain-drive test).
- Seam removes exactly one overlap per boundary: `C` clips of `N` frames emit
  `C*(N-k)` (frame-count arithmetic on the real emitter).
- Colour-match locks to clip 0, no ratchet across successively brighter clips.
- Blend monotonic in linear light, never exceeds the brighter endpoint (no
  midpoint flash); audio equal-power, no int16 wrap.
- Video/audio stay locked slice-for-slice through the seam.
- Schema + `valid_commands` render with the new `continuity` field and no command
  changes.
- All existing hard-cut tests pass unchanged.

## Honest limits — needs a GPU pass (not run on this branch; GPUs in use)

- **Continuity reproduces the bear-video chaining from one prompt** (validated in
  the fast-h3-live sibling this mode is drawn from).
- **Gap-free live is NOT GPU-validated on this branch.** It should be helped by
  the existing 256-token pad keeping continuation clips on one compiled graph,
  but it needs a GPU pass to confirm the FL2VA clip builds inside the playout
  budget at the chosen resolution.
- **FL2VA is undistilled on this checkpoint.** The seam hides the *appearance*
  discontinuity at a boundary, not the *momentum reset*: real motion continuity
  needs a distilled `transformer_ref` this checkpoint lacks. Expect motion to
  restart at each seam even though the picture no longer cuts.
- **Needs measurement on the target hardware:** FL2VA clip build time vs. playout
  budget at the chosen resolution, and long-run drift over a many-clip chain.

## Not done here (per the alignment)

- No runtime command to toggle continuity — deployment config only.
- PR intentionally not opened; Zhuma will open it.
