"""The checked-in client contract stays complete and reproducible."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_checked_in_schema_matches_runtime_render(tmp_path):
    output = tmp_path / "schema.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "reactor_runtime.schema",
            "--path",
            str(ROOT),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
        capture_output=True,
    )
    assert json.loads(output.read_text()) == json.loads(
        (ROOT / "schema.json").read_text()
    )


def test_every_command_input_and_model_message_has_contract_descriptions():
    schema = json.loads((ROOT / "schema.json").read_text())
    for path, route in schema["paths"].items():
        if not path.startswith("/events/"):
            continue
        operation = route["post"]
        assert operation["summary"], path
        body = operation.get("requestBody", {}).get("content", {})
        properties = (
            body.get("application/json", {}).get("schema", {}).get("properties", {})
        )
        for name, field in properties.items():
            assert field.get("description"), (path, name)
    for name in (
        "StateUpdate",
        "InputAccepted",
        "TakeChanged",
        "ChunkComplete",
        "GenerationEnded",
    ):
        message = schema["components"]["schemas"][name]
        wire_name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        assert schema["webhooks"][wire_name]["post"]["summary"].startswith(
            "Emitted "
        ), name
        for field_name, field in message["properties"].items():
            assert field.get("description"), (name, field_name)
