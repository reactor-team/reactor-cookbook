"""Exercise the real application hooks and model bookkeeping without a GPU."""

import asyncio
import io
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

from solarrenderer import SolarRenderer
from solarrenderer_media import validate_reference, validate_structure
from solarrenderer_model import SolarAnchor, SolarInput, SolarModel, SolarResult
from solarrenderer_types import SolarState


def app():
    model = SolarRenderer()
    model.state = SolarState()
    messages = []

    async def send(message):
        messages.append(message)

    model.send = send
    model.output = SimpleNamespace(flush=lambda: None)
    return model, messages


def test_refusal_prevents_model_call():
    model, _ = app()
    model.engine = SimpleNamespace(
        generate=lambda _: pytest.fail("refused step reached model")
    )
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    with pytest.raises(CommandError, match="structure"):
        asyncio.run(model.start())


def test_anchor_acknowledgment_and_fixed_prompt():
    model, messages = app()
    model.state._structure = b"validated video"
    with pytest.raises(CommandError):
        asyncio.run(model.start())
    asyncio.run(model.set_prompt("first prompt"))
    asyncio.run(model.start())
    first = asyncio.run(model.process_input())
    assert first.anchor.prompt == "first prompt"
    assert asyncio.run(model.process_input()).anchor is first.anchor
    asyncio.run(model.set_prompt("next video"))
    assert asyncio.run(model.process_input()).anchor.prompt == "first prompt"
    result = SolarResult(
        world_id=first.world_id, chunk_index=1, frames=None, complete=False
    )
    asyncio.run(model.process_output(StepOutcome(result=result, elapsed=0.5)))
    assert asyncio.run(model.process_input()).anchor is None
    assert messages[-1].completed_chunks == 1 and messages[-1].last_chunk_seconds == 0.5
    asyncio.run(model.start())
    next_input = asyncio.run(model.process_input())
    assert next_input.world_id == first.world_id + 1
    assert next_input.anchor.prompt == "next video"


def test_generate_reads_only_input():
    model, _ = app()
    supplied = SolarInput(world_id=11)
    result = SolarResult(world_id=11, chunk_index=1, frames=None, complete=False)

    def generate(value):
        assert value is supplied
        return result

    model.engine = SimpleNamespace(generate=generate)
    model.state = None
    assert model.generate(supplied) is result


def test_failure_does_not_acknowledge_or_complete():
    model, messages = app()
    error = RuntimeError("compute failed")
    with pytest.raises(RuntimeError, match="compute failed"):
        asyncio.run(model.process_output(StepOutcome(error=error, elapsed=0.1)))
    assert model.state._applied_id is None and model.state._chunks == 0
    assert not messages


def test_pause_completion_and_session_reset():
    model, _ = app()
    model.state._structure = b"video"
    model.state.prompt = "scene"
    asyncio.run(model.start())
    asyncio.run(model.set_paused(True))
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    asyncio.run(model.set_paused(False))
    result = SolarResult(
        world_id=1,
        chunk_index=10,
        frames=np.zeros((5, 2, 2, 3), np.uint8),
        complete=True,
    )
    output = asyncio.run(model.process_output(StepOutcome(result=result, elapsed=0.2)))
    assert output.main_video.shape == (5, 2, 2, 3)
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    calls = []
    model.engine = SimpleNamespace(reset=lambda: calls.append("reset"))
    model.on_session_ended()
    assert calls == ["reset"]
    assert not hasattr(SolarRenderer, "on_disconnected")
    assert SolarState().seed == 0 and SolarState()._anchor is None


@pytest.mark.parametrize(
    "command,mime", [("set_structure", "video/mp4"), ("set_image", "image/png")]
)
def test_bad_upload_refused(command, mime):
    model, messages = app()
    with pytest.raises(CommandError):
        asyncio.run(
            getattr(model, command)(
                UploadedFile(name="bad", mime_type=mime, data=b"bad")
            )
        )
    assert model.state._structure is None and model.state._reference is None
    assert not messages


def test_model_bookkeeping_and_failure():
    backend = SimpleNamespace(
        begin=lambda _: None, reset=lambda: None, step=lambda: (None, False)
    )
    model = SolarModel()
    model._backend = backend
    with pytest.raises(ValueError):
        model.generate(SolarInput(world_id=1))
    anchor = SolarAnchor(structure=b"video", reference=None, prompt="scene", seed=7)
    assert model.generate(SolarInput(world_id=1, anchor=anchor)).chunk_index == 1

    def fail():
        raise RuntimeError("failed")

    backend.step = fail
    with pytest.raises(RuntimeError):
        model.generate(SolarInput(world_id=1))
    backend.step = lambda: (None, True)
    assert model.generate(SolarInput(world_id=1)).chunk_index == 2
    with pytest.raises(RuntimeError, match="complete"):
        model.generate(SolarInput(world_id=1))
    model.reset()
    assert model.generate(SolarInput(world_id=2, anchor=anchor)).chunk_index == 1


def test_model_imports_without_runtime():
    script = """
import sys
class BlockRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("reactor_runtime"):
            raise RuntimeError("model imported Runtime")
sys.meta_path.insert(0, BlockRuntime())
import solarrenderer_model
import solarrenderer_backend
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_four_gpu_runner_configuration(monkeypatch, tmp_path):
    import solarrenderer
    from solarrenderer_model import SolarModel

    calls = {}
    settings = object()
    monkeypatch.setattr(solarrenderer, "get_weights_path", lambda: str(tmp_path))
    monkeypatch.setattr(solarrenderer, "read_settings", lambda *_: settings)

    class Runner:
        def __init__(self, model, **kwargs):
            calls.update(model=model, **kwargs)

        def start(self):
            calls["started"] = True

    monkeypatch.setattr(solarrenderer, "DistributedRunner", Runner)
    SolarRenderer().load(tmp_path / "solarrenderer.yaml")
    assert calls["model"] is SolarModel
    assert calls["world_size"] == 4
    assert calls["init_process_group"] is True
    assert calls["load_kwargs"] == {"settings": settings}
    assert calls["started"]


def test_four_model_uses_runner_rank_context(monkeypatch):
    import solarrenderer_backend as backend

    calls = []
    monkeypatch.setattr(
        backend,
        "FourBackend",
        lambda *args: calls.append(args) or SimpleNamespace(reset=lambda: None),
    )
    model = SolarModel()
    model.device, model.rank, model.world_size = "cuda:2", 2, 4
    settings = object()
    model.load(settings)
    assert calls == [(settings, "cuda:2", 2, 4)]
    model.reset()


def test_validated_reference_is_only_staged():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    data = buffer.getvalue()
    model, _ = app()
    asyncio.run(
        model.set_image(UploadedFile(name="red.png", mime_type="image/png", data=data))
    )
    assert model.state._reference == data
    assert not model.state._active
    asyncio.run(model.clear_image())
    assert model.state._reference is None


@pytest.mark.parametrize("frames,valid", [(5, False), (158, True), (170, True)])
def test_video_minimum_and_bounded_decode(frames, valid):
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((16, 16, 3), np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if valid:
        validate_structure(buffer.getvalue())
    else:
        with pytest.raises(ValueError, match="158"):
            validate_structure(buffer.getvalue())


def test_image_pixel_limit():
    buffer = io.BytesIO()
    Image.new("1", (4097, 4096)).save(buffer, format="PNG")
    with pytest.raises(ValueError, match="megapixel"):
        validate_reference(buffer.getvalue())


@pytest.mark.parametrize("complete", [False, True])
def test_replaced_world_result_never_mutates_new_video(complete):
    model, messages = app()
    model.state._structure, model.state.prompt = b"video", "scene"
    asyncio.run(model.start())
    old = asyncio.run(model.process_input())
    asyncio.run(model.start())
    before = len(messages)
    output = asyncio.run(
        model.process_output(
            StepOutcome(
                result=SolarResult(
                    world_id=old.world_id,
                    chunk_index=10,
                    frames=np.ones((5, 2, 2, 3), np.uint8),
                    complete=complete,
                ),
                elapsed=2,
            )
        )
    )
    assert output is None and len(messages) == before
    assert model.state._active and not model.state._complete
    assert model.state._chunks == model.state._frames == 0
    assert asyncio.run(model.process_input()).anchor is not None


def test_stop_flushes_resets_and_drops_inflight_result():
    model, messages = app()
    calls = []
    model.engine = SimpleNamespace(reset=lambda: calls.append("reset"))
    model.output = SimpleNamespace(flush=lambda: calls.append("flush"))
    model.state._structure, model.state.prompt = b"video", "scene"
    asyncio.run(model.start())
    old = asyncio.run(model.process_input())
    result = asyncio.run(model.stop())
    assert not result.active and calls[-2:] == ["reset", "flush"]
    assert model.state._structure == b"video" and model.state.prompt == "scene"
    count = len(messages)
    assert (
        asyncio.run(
            model.process_output(
                StepOutcome(
                    result=SolarResult(
                        world_id=old.world_id,
                        chunk_index=1,
                        frames=None,
                        complete=False,
                    ),
                    elapsed=1,
                )
            )
        )
        is None
    )
    assert len(messages) == count
    with pytest.raises(ApplicationError):
        asyncio.run(model.process_input())
    asyncio.run(model.start())
    assert asyncio.run(model.process_input()).world_id == old.world_id + 1


@pytest.mark.parametrize("paused", [True, False])
def test_pause_without_active_video_refuses_without_broadcast(paused):
    model, messages = app()
    with pytest.raises(CommandError) as exc:
        asyncio.run(model.set_paused(paused))
    assert exc.value.code == "not_running" and not messages


def test_join_snapshot_is_only_sent_to_joiner():
    model, messages = app()
    received = []

    async def send(status):
        received.append(status)

    asyncio.run(model.on_connected(SimpleNamespace(send=send)))
    assert len(received) == 1 and not messages


def test_backend_loads_weights_once_and_uses_temporary_session_files(
    monkeypatch, tmp_path
):
    import importlib.util
    from collections import Counter
    from contextlib import nullcontext

    from solarrenderer_backend import FourBackend
    from solarrenderer_model import SolarSettings

    calls, members = Counter(), []
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/render.py").touch()
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "modular_model_index.json").touch()
    for name in ["control", "fewstep"]:
        (weights / name).touch()
    settings = SolarSettings(
        source=source,
        h3=weights,
        control=weights / "control",
        fewstep=weights / "fewstep",
    )

    def mark(name):
        return lambda *args: calls.update([name])

    def load_encoder(*args):
        calls["encoder"] += 1
        return object()

    def load_renderer(*args, **kwargs):
        calls["renderer"] += 1
        return object(), SimpleNamespace(video_vae=object()), "cuda:0"

    def encode(encoder, sample, path):
        assert encoder is not None and Path(path).parent.is_dir()
        calls["encode"] += 1
        members.append(path)
        Path(path).write_bytes(b"encoded")

    native = SimpleNamespace(
        load_encoder=load_encoder,
        load_renderer=load_renderer,
        encode_sample=encode,
        sample_of=lambda *args: args,
    )
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda spec: native)
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda *args: SimpleNamespace(
            loader=SimpleNamespace(exec_module=lambda module: None)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "solarwm.backends.minimax_h3.distributed",
        SimpleNamespace(init_sequence_parallel=mark("group")),
    )
    monkeypatch.setitem(
        sys.modules,
        "solarrenderer_optimizations",
        SimpleNamespace(
            cache_text_refiner=mark("text"),
            install_fa4=mark("fa4"),
            precompute_adaln=mark("adaln"),
            StaticPrefixRun=lambda *args: SimpleNamespace(frames=40),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "solarrenderer_vae", SimpleNamespace(install=mark("vae"))
    )
    monkeypatch.setitem(
        sys.modules,
        "solarrenderer_decode",
        SimpleNamespace(SolarDecoder=lambda *args: object()),
    )
    dist = SimpleNamespace(
        broadcast_object_list=lambda *args, **kwargs: None,
        get_world_size=lambda: 4,
        all_gather_object=lambda *args: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            distributed=dist,
            manual_seed=lambda seed: None,
            no_grad=nullcontext,
            Generator=lambda **kwargs: SimpleNamespace(
                manual_seed=lambda seed: object()
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    class Stream:
        def read(self, record):
            assert Path(record["member"]).exists()
            return object()

    monkeypatch.setitem(
        sys.modules,
        "solarwm.backends.minimax_h3.omni_data",
        SimpleNamespace(H3OmniStream=Stream),
    )
    backend = FourBackend(settings, "cuda:0", 0, 4)
    anchor = SolarAnchor(structure=b"video", reference=None, prompt="scene", seed=0)
    backend.begin(anchor)
    backend.reset()
    backend.begin(anchor)
    assert calls == Counter(
        encoder=1, renderer=1, encode=2, group=1, text=1, fa4=1, adaln=1, vae=1
    )
    assert all(
        not Path(member).exists() and not Path(member).is_relative_to(weights)
        for member in members
    )
