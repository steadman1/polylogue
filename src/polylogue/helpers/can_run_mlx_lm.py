import platform
import sys

# written by Gemini
def can_run_mlx_lm() -> bool:
    """Verifies OS, Apple Silicon architecture, and MLX Metal device availability."""
    # 1. Check OS is macOS
    if sys.platform != "darwin":
        return False

    # 2. Check architecture is Apple Silicon (arm64 / aarch64)
    machine = platform.machine().lower()
    if machine not in ("arm64", "aarch64"):
        return False

    # 3. Check MLX / MLX-LM imports and Metal GPU backend availability
    try:
        import mlx.core as mx

        # Verify the Metal backend is functional and default device is available
        return mx.metal.is_available()
    except (ImportError, AttributeError):
        return False
