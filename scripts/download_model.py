#!/usr/bin/env python3
"""Pre-download Whisper model for offline deployment.

This script downloads the Whisper model to a local directory before
first run, enabling offline deployment on air-gapped Windows machines.

Usage:
    # Download to default HuggingFace cache (~3GB)
    python scripts/download_model.py

    # Download to specific directory (recommended for offline deployment)
    python scripts/download_model.py --output D:\Models\whisper

    # Specify model size (default: large-v3-turbo)
    python scripts/download_model.py --model large-v3-turbo --output D:\Models\whisper

Environment variables:
    HF_HOME: Override HuggingFace cache directory.
    WHISPER_MODEL: Model size to download (default: large-v3-turbo).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from faster_whisper import WhisperModel


def get_model_dir(output: str | None, model_size: str) -> Path:
    """Determine the model download directory.

    Args:
        output: Explicit output directory, or None for default.
        model_size: Whisper model size.

    Returns:
        Path to the model directory.
    """
    if output:
        base_dir = Path(output).expanduser().resolve()
        return base_dir / model_size

    # Default to HuggingFace cache subdirectory
    hf_home = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    return Path(hf_home) / "hub" / f"models--Systran--faster-whisper-{model_size}"


def download_model(
    model_size: str = "large-v3-turbo",
    output: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
) -> Path:
    """Download and cache the Whisper model.

    Args:
        model_size: Whisper model size (tiny, base, small, medium, large-v3, large-v3-turbo).
        output: Optional local directory to save model files.
        device: Device for model loading (cpu, cuda).
        compute_type: Compute type (int8, float16, etc.).

    Returns:
        Path to the downloaded model directory.
    """
    model_dir = get_model_dir(output, model_size)
    print(f"[INFO] Model: {model_size}")
    print(f"[INFO] Download destination: {model_dir}")

    # Set HuggingFace cache if using default location
    if output is None:
        hf_home = os.environ.get(
            "HF_HOME", str(Path.home() / ".cache" / "huggingface")
        )
        os.environ["HF_HOME"] = hf_home
        os.environ["TRANSFORMERS_CACHE"] = hf_home
        print(f"[INFO] HuggingFace cache: {hf_home}")
    else:
        print(f"[INFO] Model will be saved to: {model_dir}")

    print("[INFO] Starting download (first run: ~3GB, may take 10-30 minutes)...")
    print("[INFO] Press Ctrl+C to cancel")

    try:
        # Load model (triggers download if not cached)
        # faster-whisper will download to HF_HOME if model_size is not a local path
        model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
        )

        print(f"[OK] Model '{model_size}' downloaded and verified successfully!")
        print(f"[INFO] Model location: {model_dir}")
        print(f"[INFO] Device: {device}, Compute type: {compute_type}")

        return model_dir

    except KeyboardInterrupt:
        print("\n[WARN] Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        sys.exit(1)


def verify_model(model_path: str, model_size: str = "large-v3-turbo") -> bool:
    """Verify that a locally cached model is valid.

    Args:
        model_path: Path to the model directory.
        model_size: Model size that was downloaded.

    Returns:
        True if model is valid, False otherwise.
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"[ERROR] Model directory does not exist: {model_dir}")
        return False

    # Check for expected model files
    # faster-whisper CTranslate2 models have these files
    expected_files = ["config.json", "model.bin"]
    missing = [f for f in expected_files if not (model_dir / f).exists()]

    if missing:
        print(f"[WARN] Model directory missing files: {missing}")
        print(f"[WARN] Expected files: {expected_files}")
        return False

    print(f"[OK] Model directory looks valid: {model_dir}")
    return True


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-download Whisper model for offline deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        "--model-size",
        dest="model_size",
        default=os.environ.get("WHISPER_MODEL", "large-v3-turbo"),
        help="Whisper model size (tiny, base, small, medium, large-v3, large-v3-turbo). "
        "Default: large-v3-turbo",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        default=None,
        help="Local directory to save model files. "
        "Default: HuggingFace cache (~/.cache/huggingface/hub/)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model loading. Default: cpu (GPU not needed for download)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type for model. Default: int8 (CPU-friendly)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing local model instead of downloading",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        help="Path to verify (use with --verify)",
    )

    args = parser.parse_args()

    if args.verify:
        if not args.model_path:
            print("[ERROR] --model-path required with --verify")
            sys.exit(1)
        success = verify_model(args.model_path, args.model_size)
        sys.exit(0 if success else 1)
    else:
        download_model(
            model_size=args.model_size,
            output=args.output,
            device=args.device,
            compute_type=args.compute_type,
        )


if __name__ == "__main__":
    main()
