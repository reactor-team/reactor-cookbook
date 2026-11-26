# Vendored: RLWRLD/RLDX-1

- Source:  https://github.com/RLWRLD/RLDX-1
- Commit:  ecbfaf80cd031dcc892186ed30465de3591047e6
- License: Apache-2.0

The `rldx` package is the inference and optimized-serving subset used by this
recipe. Training, fine-tuning, and evaluation-only files are not included. The
repository root `LICENSE` contains the Apache-2.0 license.

Three vendored files contain the Reactor RTC integration:

- `data/state_action/state_action_processor.py` accepts physical-unit action
  prefixes at the policy boundary;
- `model/modules/action_model/rtc.py` supports guided RTC inference; and
- `policy/policy_runtime.py` passes the RTC request into the model.

Three vendored files load the policy only from local folders:

- `policy/policy_loader.py` requires a local checkpoint folder and raises
  `FileNotFoundError` for a missing one instead of calling
  `snapshot_download`. Its optional `base_model_path` replaces the base VLM
  id that the checkpoint names (`model_name`) in the model and processor
  loads, so the backbone config and the tokenizer load from disk;
- `policy/rldx_policy.py` passes `base_model_path` to the loader; and
- `model/core/processing_rldx.py` accepts `model_name` as a
  `from_pretrained` override.

All other differences from the source commit are file pruning.
