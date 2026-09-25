# Porting Cosmos3-Policy-DROID onto the step loop

What the model was, what the two halves are, and every decision that was not
a mechanical translation. The rules this port follows are the
`application-model-isolation` and `porting-to-reactor-app` skills in the
[reactor-runtime](https://github.com/reactor-team/reactor-runtime) repository.

## What the model was

A `ReactorPipeline` on reactor-runtime 3.2.6 serving Cosmos3-Nano-Policy-DROID.
`inference()` was a `while True` that yielded `Idle` every turn, sampled the
three input tracks and three state strings into a dict, handed it to a
`CosmosDroidRealtimeAdapter` that owned frame retention, both gates, the
proprio parsing, and the session bookkeeping, and sent an `ActionPrediction`
for each chunk the adapter returned. The adapter wrapped a stateless
`CosmosDroidPolicy` around upstream's `RobolabPolicyService`, served from a
232-file pruned copy of `cosmos_framework` checked into that repository.
`reset` detached and re-attached the adapter's session.

The Edge checkpoint was not served anywhere.

## The two halves

`cosmos3_policy_droid_model.py` (model): `Cosmos3PolicyModel` with
`load(config_path, weights_root)`, `generate(input)`, `reset()`. It owns the
upstream policy service, its load-time wrapping, the warmup, and the shape
check. `PolicyInput` carries three frames, the joint and gripper arrays, and
the task; `PolicyResult` carries the `(32, 8)` chunk.

`cosmos3_policy_droid.py` (application): `Cosmos3PolicyDroid(ReactorApp)` with
`process_input()` (frame retention and both gates), the one-line `generate()`,
`process_output()` (the step counter and the message), the `reset` command,
and the session hooks. `cosmos3_policy_droid_types.py` keeps the client
contract.

The rendered schema is identical to the fleet model's except for the title
and the description.

## Decisions

1. **The gates are refusals.** The adapter's "first-prediction gate" (every
   view has a frame, proprio parses) and "advance gate" (the echoed step
   strictly increased) were `continue` statements in a tick loop. Each is an
   `ApplicationError` in `process_input()`. The advance gate's bookkeeping
   (`_last_executed`) moves at the moment the gate opens, as the adapter
   did; a failure in `generate()` after that ends the session, so no
   half-open state survives it.

2. **Frame retention stays on the application.** Cameras deliver
   asynchronously and a step reads each track once, so a prediction is built
   from the newest frame per view kept across steps. Which frames a client
   sent is a client fact; the dict lives on the app, is emptied at session
   start, on `reset`, and at session end, and never touches the model.

3. **The step counter stays on the application.** The skill sends counters
   to the model, which counts its own steps. This policy is stateless and
   counts nothing; `step` is the session's flow-control number the client
   echoes and `reset` zeroes. It is a private state field the runtime
   re-creates per session.

4. **`reset()` on the model half is empty and still exists.** The contract
   requires it and `@session_ended` calls it. The policy has nothing to
   forget; the docstring says so rather than inventing work.

5. **The gate is kept as deployed.** A request/reply contract keyed on a
   client `chunk_id` (the shape `docs/robot-policy-client-contract.md` in
   reactor-models describes) is planned as its own contract change. This
   port moves the runtime without moving the schema.

6. **Proprio must be finite.** The adapter accepted any float. `parse_proprio`
   now also rejects NaN or infinite values, since a fabricated state would
   command a real arm. A rejected value reads as "no proprio yet" and refuses
   the step, the same as a malformed one.

7. **Both checkpoints, one code path.** Upstream serves Edge and Nano through
   the same `RobolabPolicyService` with three arguments changed
   (`checkpoint_path`, `format_prompt_as_json`, `guidance_interval`). They
   are config keys read by the model half; the default is Edge.

8. **Weights come from Hugging Face through `huggingface_hub`.** Upstream's
   `checkpoint_db` downloads by shelling out to a separate `uv` project and
   the `hf` CLI. `route_checkpoint_downloads` replaces both of its
   resolvers with `snapshot_download` / `hf_hub_download` into the weights
   directory the application resolves once with `get_weights_path()` and
   hands to `load()`. The fleet model instead read a curator-staged
   directory with Hugging Face offline; the cookbook has no curator.

9. **The guardrail wrapper passes its arguments through.** Upstream's
   `_build_setup_args` gained a `parallelism_overrides` argument at the
   pinned commit; the wrapper takes `*args, **kwargs` so the next upstream
   signature change does not break it either.

10. **Warmup is model code.** Two throwaway predictions at load pay
    `torch.compile`'s first call (about 20 s on a B200) before the first
    session, and the second confirms the compiled path (about 210 ms for
    Edge). `warmup: false` in the config skips it.

11. **The upstream source is cloned, not vendored.** The fleet model checks a
    pruned 232-file copy of `cosmos_framework` into its repository. Here,
    as for the other cookbook models, `cosmos3_policy_droid.yaml` pins the
    public repository and a full commit hash (`cf5d68c`, upstream's
    2026-09-23 release), and the model half clones it blobless into the
    weights root on first load, checks it out detached at that commit,
    refuses a checkout at another revision or with local changes, and puts
    it first on `sys.path` before importing. Nothing under the checkout is
    edited; the two behaviours the port changes (guardrails, downloads) are
    applied by wrapping at load. Bumping upstream is one hash in the config.

12. **`requirements.txt` is measured, not copied.** The full upstream
    package declares training and serving dependencies together. The pins
    here are the distributions the policy serving path imported during a
    live `sys.modules` capture on a B200 with `COSMOS_TRAINING=0`, at the
    versions upstream's lock resolves for its cu128 group.

13. **wandb is not a dependency.** Upstream imports it only inside
    two training-loop functions (a trainer timeout handler and a straggler
    report), and its release pins conflict with the runtime's protobuf.
    Nothing on the serving path reaches either function.

## Verification

- `PYTHONPATH=. python -m pytest tests/ -q` (30 tests, no GPU): the contract,
  every refusal, the echo gate across steps, frame retention, the message
  and its `step`, the error branch, `reset`, session end, the parsers, the
  config, the pinned checkout against a local Git repository (clone at the
  pin, drift and local edits refused), and the model half with the
  framework stubbed.
- `python -m reactor_runtime.schema --path .` diffed against the fleet
  model's: title and description only.
- The image built by `reactor build` cloned the source and served the Edge
  checkpoint on one B200: load 6 s, warmup 20 s, 254 ms per chunk; a
  `reactor-sdk` client saw both gates hold and a 277 ms round trip.
