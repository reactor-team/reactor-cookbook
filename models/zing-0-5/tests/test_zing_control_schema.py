"""Native key semantics are explicit, never inferred from external action labels."""

from reactor_runtime.interface.model.contract import ModelContract
from zing import Zing


def test_each_native_key_has_its_own_exact_direction():
    """A schema-only mapper can distinguish native j from an external down label."""
    contract = ModelContract.of(Zing)
    field = contract.commands["set_key"].command.__command_fields__["key"]
    description = field.info.description
    expected = {
        "w": "move forward",
        "a": "strafe left",
        "s": "move backward",
        "d": "strafe right",
        "i": "look up",
        "j": "look left",
        "k": "look down",
        "l": "look right",
    }
    for key, meaning in expected.items():
        assert f"`{key}`: {meaning}" in description
    assert "`j`: look down" not in description
    assert "Looking down requires native `k`, never native `j`" in description
