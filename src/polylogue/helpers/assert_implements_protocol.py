import inspect
from typing import get_type_hints


def assert_implements_protocol(target_cls: type, protocol_cls: type) -> None:
    """Verifies that target_cls implements all protocol methods with matching signatures."""
    for name, proto_attr in inspect.getmembers(
        protocol_cls, predicate=inspect.isfunction
    ):
        # Ignore private / dunder protocol internals
        if name.startswith("_"):
            continue

        assert hasattr(target_cls, name), f"Missing method: {name}"

        proto_sig = inspect.signature(proto_attr)
        target_sig = inspect.signature(getattr(target_cls, name))

        assert proto_sig == target_sig, (
            f"Signature mismatch for method '{name}':\n"
            f"  Expected: {proto_sig}\n"
            f"  Actual:   {target_sig}"
        )
