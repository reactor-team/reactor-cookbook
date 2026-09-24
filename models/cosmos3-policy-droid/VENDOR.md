# Vendored code provenance

`cosmos_framework/` is a pruned copy of the upstream package from
https://github.com/NVIDIA/cosmos-framework at commit
`cf5d68c` (`Release 2026-09-23`), vendored on 2026-09-23.

- Code license: OpenMDW-1.1. `LICENSE.cosmos-framework` and
  `NOTICE.cosmos-framework` are upstream's `LICENSE` and `NOTICE`, verbatim.
  Every copied source file keeps its upstream SPDX header.
- Model weights (`nvidia/Cosmos3-Edge-Policy-DROID`,
  `nvidia/Cosmos3-Nano-Policy-DROID`, and the `Wan-AI/Wan2.2-TI2V-5B` video
  tokenizer they load) are downloaded from Hugging Face at first load and are
  never checked in.
- Local modifications: **none**. Every file under `cosmos_framework/` is
  byte-identical to upstream at the pinned commit. Behaviour the port needs
  to change is applied by wrapping at load time in
  `cosmos3_policy_droid_model.py`; nothing under `cosmos_framework/` is
  edited, so a re-vendor is a copy, not a merge.

## What is kept

Upstream is a training and inference framework of 883 Python files. This
folder serves one path through it: `RobolabPolicyService.infer()` in
`cosmos_framework/scripts/action_policy_server_robolab.py`, the DROID
policy server. The keep-set is the union of five layers:

1. **The measured import closure.** `sys.modules` captured on a B200 after a
   real `RobolabPolicyService` load and six `infer()` calls on the
   Edge-Policy-DROID checkpoint, with `COSMOS_TRAINING=0` and all three
   attention kernels (`flash-attn`, `flash-attn-3-nv`, `natten`) installed:
   244 dotted module names, listed in `cosmos_framework.closure.txt`. Static
   analysis is not trusted for this codebase (lazy imports, `TYPE_CHECKING`
   guards, config-driven dispatch); the closure is measured, not inferred.
2. **Package `__init__.py` chains** for every kept module. Six directories
   on those paths are upstream PEP 420 namespace packages with no
   `__init__.py` (`configs`, `data/imaginaire`,
   `data/imaginaire/webdataset`, `data/imaginaire/webdataset/augmentors`,
   `utils/env_parsers`, `utils/functional`); they exist once their children
   are copied.
3. **Data files kept modules load by path.** Seven reasoner HF-style config
   JSONs referenced from `configs/base/defaults/reasoner.py`; the three
   `inference/configs/model/Cosmos3-*.yaml` checkpoint configs;
   `inference/defaults/wam/sample_args.json` and
   `inference/defaults/neg_prompts.json` (the policy server samples in
   `wam` mode); and the two action normalizer stats JSONs
   `data/generator/action/utils/action_processing.py` points at through
   `Path(__file__)`. String literals that name checkpoint files
   (`config.json`, `tokenizer.json`, ...) or training data are not
   in-tree files and are left out.
4. **Both branches of every attention backend probe.** Each backend's
   `__init__.py` imports `functions` when its kernel package is present and
   `stubs` when it is absent. The capture exercised one side per backend
   (cuDNN attention was unsupported in the capture environment); both sides
   are kept so import succeeds whatever the image carries.
5. **`except ImportError` fallbacks** inside kept modules
   (`utils/one_logger/one_logger_override_utils.py`).

Result: 257 files (243 `.py`, 14 data), 4.4 MB. Excluded by construction:
training code, datasets, callbacks, evaluation, the WebSocket server path
(`openpi-server`), the `utils/hf_cli` uv project the upstream downloader
shells out to (this port downloads through `huggingface_hub` instead), and
every `*_test.py`.

## Verification

- Every entry in `cosmos_framework.closure.txt` resolves in this tree.
- The smoke test that produced the closure was re-run with `PYTHONPATH`
  pointing at this tree instead of the upstream checkout: all 244 modules
  loaded from the vendored copy, and the predicted `[32, 8]` chunk was
  bit-identical to the upstream run.
- `diff -rq` against upstream at the pinned commit reports no content
  differences for any kept file.

## Re-vendoring

1. Clone upstream at the new commit and build its environment
   (`uv sync --all-extras --group=cu128 --group=policy-server`).
2. Load the policy through `RobolabPolicyService`, run several `infer()`
   calls, and dump `sorted(m for m in sys.modules if m.startswith("cosmos_framework"))`.
3. Re-derive layers 2 to 5 above and copy the files. Diff the new closure
   against `cosmos_framework.closure.txt` before assuming nothing moved.
4. Re-run the smoke test against the vendored tree and check that no module
   loads from outside it.
5. Update the commit and date at the top of this file.
