"""Reuse the pinned upstream encoder and renderer inside an isolated model process."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

from solarrenderer_model import SolarAnchor, SolarSettings


class FourBackend:
    def __init__(self, settings: SolarSettings, device, rank, world_size):
        self.settings = settings
        for path in (
            settings.source / "scripts/render.py",
            settings.h3 / "modular_model_index.json",
            settings.control,
            settings.fewstep,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing SolarRenderer asset: {path}")
        sys.path[:0] = [str(settings.source / "src"), str(settings.source / "tools")]
        spec = importlib.util.spec_from_file_location(
            "solarrenderer_native_render", settings.source / "scripts/render.py"
        )
        self.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.native)
        self.renderer = self.run = self.decoder = None
        from solarwm.backends.minimax_h3.distributed import init_sequence_parallel

        from solarrenderer_optimizations import (
            cache_text_refiner,
            install_fa4,
            precompute_adaln,
        )
        from solarrenderer_vae import install

        self.device, self.rank = device, rank
        init_sequence_parallel(world_size)
        self.encoder = (
            self.native.load_encoder(str(settings.h3), device) if rank == 0 else None
        )
        self.renderer = self.native.load_renderer(
            str(settings.h3),
            str(settings.control),
            str(settings.fewstep),
            device,
            causal=True,
        )
        core, conditioners, _ = self.renderer
        precompute_adaln(core)
        cache_text_refiner(core)
        install_fa4()
        install(conditioners.video_vae)

    def reset(self):
        self.run = self.decoder = None

    def begin(self, anchor: SolarAnchor):
        import torch
        import torch.distributed as dist
        from solarwm.backends.minimax_h3.omni_data import H3OmniStream

        from solarrenderer_decode import SolarDecoder
        from solarrenderer_optimizations import StaticPrefixRun

        self.reset()
        directory = None
        result = [None]
        if self.rank == 0:
            try:
                directory = tempfile.TemporaryDirectory(prefix="solar-sp4-")
                root = Path(directory.name)
                structure, reference, member = (
                    root / "structure.mp4",
                    root / "first.png",
                    root / "member.safetensors",
                )
                structure.write_bytes(anchor.structure)
                if anchor.reference is not None:
                    reference.write_bytes(anchor.reference)
                sample = self.native.sample_of(
                    "reactor",
                    anchor.prompt,
                    str(structure),
                    str(reference) if anchor.reference is not None else None,
                )
                self.native.encode_sample(self.encoder, sample, str(member))
                result[0] = {"member": str(member)}
            except Exception as exc:  # noqa: BLE001 - propagate preparation failures to every rank
                result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(result, src=0)
        if "error" in result[0]:
            if directory is not None:
                directory.cleanup()
            raise RuntimeError(result[0]["error"])
        stream = object.__new__(H3OmniStream)
        stream.root, stream.current_plan_fingerprint, stream.aligned_control = (
            "/",
            "render",
            False,
        )
        batch = None
        read_error = None
        try:
            batch = stream.read({"id": "reactor", "member": result[0]["member"]})
        except Exception as exc:  # noqa: BLE001 - every rank must reach the failure exchange
            read_error = f"Rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, read_error)
        if directory is not None:
            directory.cleanup()
        if any(error is not None for error in errors):
            raise RuntimeError(
                "; ".join(error for error in errors if error is not None)
            )
        core, conditioners, device = self.renderer
        torch.manual_seed(anchor.seed)
        with torch.no_grad():
            self.run = StaticPrefixRun(
                core,
                batch,
                torch.Generator(device=device).manual_seed(anchor.seed),
            )
        self.decoder = SolarDecoder(conditioners.video_vae, self.run.frames, device)

    def step(self):
        import torch

        block = self.run.step()
        decoded = self.decoder.push(block)
        frames = None
        if decoded is not None:
            mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
            std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
            pixels = (decoded.float() * std + mean).clamp(0, 1)
            frames = (
                pixels[0]
                .permute(1, 2, 3, 0)
                .mul(255)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .numpy()
            )
        return frames, self.run.complete
