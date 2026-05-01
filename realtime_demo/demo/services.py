from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Optional
from uuid import uuid4

from django.conf import settings


DEFAULT_CLASS_NAMES = ("bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ")
REQUIRED_RUNTIME_MODULES = ("torch", "torchaudio", "soundfile")


@dataclass(frozen=True)
class DemoPaths:
    config_path: str
    ckpt_path: str
    metadata_path: Optional[str]
    conditioning_mode: Optional[str]
    class_names: tuple[str, ...]
    device: Optional[str]


def resolve_path(path: str) -> str:
    if not path:
        return ""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((settings.BASE_DIR / candidate).resolve())


def display_path(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(settings.BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def parse_class_names(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def infer_class_names_from_metadata(metadata_path: Optional[str]) -> tuple[str, ...]:
    if not metadata_path:
        return ()
    path = Path(metadata_path)
    if not path.exists():
        return ()

    names = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            class_name = row.get("class")
            if class_name is not None:
                names.add(str(class_name))
    return tuple(sorted(names))


def resolve_class_names(metadata_path: Optional[str], requested: str = "") -> tuple[str, ...]:
    return (
        parse_class_names(requested)
        or parse_class_names(settings.TAD_CLASS_NAMES)
        or infer_class_names_from_metadata(metadata_path)
        or DEFAULT_CLASS_NAMES
    )


def default_paths() -> DemoPaths:
    metadata_path = settings.TAD_METADATA_PATH or None
    conditioning_mode = settings.TAD_CONDITIONING_MODE or None
    device = settings.TAD_DEVICE or None
    ckpt_path = settings.TAD_CKPT_PATH
    if not ckpt_path:
        candidates = checkpoint_candidates(limit=1)
        ckpt_path = candidates[0] if candidates else ""
    return DemoPaths(
        config_path=resolve_path(settings.TAD_CONFIG_PATH),
        ckpt_path=resolve_path(ckpt_path),
        metadata_path=resolve_path(metadata_path) if metadata_path else None,
        conditioning_mode=conditioning_mode,
        class_names=resolve_class_names(resolve_path(metadata_path) if metadata_path else None),
        device=device,
    )


def request_paths(
    *,
    config_path: str = "",
    ckpt_path: str = "",
    metadata_path: str = "",
    conditioning_mode: str = "",
    class_names: str = "",
) -> DemoPaths:
    defaults = default_paths()
    resolved_metadata_path = resolve_path(metadata_path) if metadata_path else defaults.metadata_path
    return DemoPaths(
        config_path=resolve_path(config_path) if config_path else defaults.config_path,
        ckpt_path=resolve_path(ckpt_path) if ckpt_path else defaults.ckpt_path,
        metadata_path=resolved_metadata_path,
        conditioning_mode=conditioning_mode or defaults.conditioning_mode,
        class_names=resolve_class_names(resolved_metadata_path, class_names),
        device=defaults.device,
    )


def checkpoint_candidates(limit: int = 5) -> list[str]:
    ckpt_root = settings.BASE_DIR / "logs" / "ckpts"
    if not ckpt_root.exists():
        return []
    candidates = sorted(
        ckpt_root.rglob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [display_path(str(path)) for path in candidates[:limit]]


def validate_paths(paths: DemoPaths) -> list[str]:
    missing = []
    if not paths.config_path or not Path(paths.config_path).exists():
        missing.append("config")
    if not paths.ckpt_path or not Path(paths.ckpt_path).exists():
        missing.append("checkpoint")
    if paths.metadata_path and not Path(paths.metadata_path).exists():
        missing.append("metadata")
    return missing


def _checkpoint_signature(config_path: str) -> str:
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except Exception:
        return "unknown"

    model_target = str((config.get("model") or {}).get("_target_", ""))
    if "EmbeddingConditionalModel" in model_target:
        return "embedding_conditional"
    if "ConditionalModel" in model_target:
        return "channel_conditional"
    return "unconditional"


def _checkpoint_conditioning_dim(state_dict: dict) -> Optional[int]:
    for key in (
        "embedding_to_conditioning.bias",
        "class_embedding.weight",
        "text_embedding.weight",
    ):
        value = state_dict.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])

    for key, value in state_dict.items():
        if key.endswith(".norm_context.weight") and hasattr(value, "shape") and len(value.shape) == 1:
            return int(value.shape[0])
    return None


@lru_cache(maxsize=16)
def _checkpoint_compatibility_cached(
    config_path: str,
    ckpt_path: str,
    ckpt_mtime: float,
) -> dict:
    del ckpt_mtime
    signature = _checkpoint_signature(config_path)
    try:
        import torch

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_keys = set((checkpoint.get("state_dict") or {}).keys())
        state_dict = checkpoint.get("state_dict") or {}
        checkpoint_note = checkpoint.get("demo_note", "")
    except Exception as exc:
        return {
            "compatible": False,
            "warning": f"Could not inspect checkpoint: {exc}",
            "signature": signature,
            "note": "",
        }

    if signature == "embedding_conditional":
        markers = (
            "class_embedding.weight",
            "embedding_to_conditioning.weight",
            "text_embedding.weight",
            "text_to_latent.weight",
        )
        if not any(any(marker in key for marker in markers) for key in state_keys):
            return {
                "compatible": False,
                "warning": (
                    "The selected config is embedding-conditional, but this checkpoint "
                    "does not contain embedding or contrastive conditioning weights."
                ),
                "signature": signature,
                "note": checkpoint_note,
            }
        checkpoint_dim = _checkpoint_conditioning_dim(state_dict)
        if checkpoint_dim and checkpoint_dim != len(DEFAULT_CLASS_NAMES):
            checkpoint_note = (
                checkpoint_note
                or f"Checkpoint uses {checkpoint_dim} conditioning classes; set class names to match that order."
            )

    if signature == "channel_conditional":
        first_conv = next((key for key in state_keys if key.endswith("input_proj.weight")), "")
        if first_conv:
            try:
                import torch

                weight = torch.load(ckpt_path, map_location="cpu")["state_dict"][first_conv]
                if tuple(weight.shape)[1] < 4:
                    return {
                        "compatible": False,
                        "warning": (
                            "The selected config expects audio plus conditioning channels, "
                            "but this checkpoint appears to have only audio input channels."
                        ),
                        "signature": signature,
                        "note": checkpoint_note,
                    }
            except Exception:
                pass

    return {"compatible": True, "warning": "", "signature": signature, "note": checkpoint_note}


def checkpoint_compatibility(paths: DemoPaths) -> dict:
    if not paths.config_path or not paths.ckpt_path:
        return {"compatible": False, "warning": "", "signature": "unknown", "note": ""}
    ckpt = Path(paths.ckpt_path)
    config = Path(paths.config_path)
    if not ckpt.exists() or not config.exists():
        return {"compatible": False, "warning": "", "signature": "unknown", "note": ""}
    return _checkpoint_compatibility_cached(
        str(config),
        str(ckpt),
        ckpt.stat().st_mtime,
    )


def _runtime_status() -> tuple[str, bool, list[str]]:
    import importlib.util

    missing_modules = [
        module for module in REQUIRED_RUNTIME_MODULES if importlib.util.find_spec(module) is None
    ]
    if missing_modules:
        return "unavailable", False, missing_modules

    try:
        import torch
    except Exception:
        return "unavailable", False, ["torch"]
    return "cuda" if torch.cuda.is_available() else "cpu", torch.cuda.is_available(), []


@lru_cache(maxsize=2)
def get_context(paths: DemoPaths) -> Any:
    from main.inference_helpers import load_inference_context

    context = load_inference_context(
        paths.config_path,
        paths.ckpt_path,
        conditioning_mode_override=paths.conditioning_mode,
        metadata_path_override=paths.metadata_path,
        class_names_override=list(paths.class_names),
        device=paths.device,
    )
    context.model.eval()
    return context


def context_status() -> dict:
    paths = default_paths()
    missing = validate_paths(paths)
    detected_device, cuda_available, missing_modules = _runtime_status()
    if missing_modules:
        missing.append("runtime dependencies")
    compatibility = checkpoint_compatibility(paths)
    if not missing and not compatibility["compatible"]:
        missing.append("compatible checkpoint")
    return {
        "ready": not missing,
        "missing": missing,
        "missing_modules": missing_modules,
        "checkpoint_warning": compatibility["warning"],
        "checkpoint_signature": compatibility["signature"],
        "checkpoint_note": compatibility.get("note", ""),
        "config_path": display_path(paths.config_path),
        "ckpt_path": display_path(paths.ckpt_path),
        "ckpt_autodetected": bool(paths.ckpt_path and not settings.TAD_CKPT_PATH),
        "metadata_path": display_path(paths.metadata_path),
        "checkpoint_candidates": checkpoint_candidates(),
        "conditioning_mode": paths.conditioning_mode,
        "class_names": list(paths.class_names),
        "class_names_csv": ",".join(paths.class_names),
        "device": paths.device or detected_device,
        "cuda_available": cuda_available,
    }


def _save_uploaded_reference(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        for chunk in uploaded_file.chunks():
            handle.write(chunk)
        return handle.name


def _write_audio(sample: Any, sample_rate: int, label: str) -> str:
    import soundfile as sf
    from main.inference_helpers import prepare_audio_for_display

    output_dir = settings.MEDIA_ROOT / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{label}-{uuid4().hex[:10]}.wav"
    path = output_dir / filename
    audio = prepare_audio_for_display(sample).cpu()
    if audio.ndim == 2:
        audio_np = audio.transpose(0, 1).numpy()
    else:
        audio_np = audio.numpy()
    sf.write(path, audio_np, sample_rate)
    return f"{settings.MEDIA_URL}generated/{filename}"


def _rms(sample: Any) -> float:
    import torch
    from main.inference_helpers import prepare_audio_for_display

    audio = prepare_audio_for_display(sample)
    return float(torch.sqrt(torch.mean(audio.square()) + 1e-12).item())


def _spectral_centroid(sample: Any, sample_rate: int) -> float:
    import torch
    from main.inference_helpers import prepare_audio_for_display

    audio = prepare_audio_for_display(sample)
    mono = audio.mean(dim=0) if audio.ndim == 2 else audio
    window = torch.hann_window(min(1024, mono.numel()))
    if mono.numel() < window.numel():
        mono = torch.nn.functional.pad(mono, (0, window.numel() - mono.numel()))
    spectrum = torch.fft.rfft(mono[: window.numel()] * window)
    magnitude = spectrum.abs()
    freqs = torch.linspace(0, sample_rate / 2, magnitude.numel())
    return float(((freqs * magnitude).sum() / (magnitude.sum() + 1e-12)).item())


def run_generation(
    *,
    target_class: str,
    reference_class: str,
    reference_split: str,
    reference_index: Optional[int],
    uploaded_file,
    seed: int,
    noise_scale: float,
    steps: int,
    benchmark_steps: list[int],
    config_path: str = "",
    ckpt_path: str = "",
    metadata_path: str = "",
    conditioning_mode: str = "",
    class_names: str = "",
) -> dict:
    import torch
    from main.inference_helpers import (
        build_noised_model_input,
        generate_conditional_sample_from_input,
        load_waveform_file,
        prepare_conditioning_inputs,
        resolve_reference_waveform,
    )

    paths = request_paths(
        config_path=config_path,
        ckpt_path=ckpt_path,
        metadata_path=metadata_path,
        conditioning_mode=conditioning_mode,
        class_names=class_names,
    )
    if target_class not in paths.class_names:
        raise ValueError(f"Unknown target class: {target_class}. Expected one of {list(paths.class_names)}")
    if reference_class not in paths.class_names:
        raise ValueError(f"Unknown reference class: {reference_class}. Expected one of {list(paths.class_names)}")
    missing = validate_paths(paths)
    if missing:
        raise FileNotFoundError(
            "Missing demo asset(s): "
            + ", ".join(missing)
            + ". Set TAD_CONFIG_PATH, TAD_CKPT_PATH, and optionally TAD_METADATA_PATH."
        )
    compatibility = checkpoint_compatibility(paths)
    if not compatibility["compatible"]:
        raise ValueError(
            compatibility["warning"]
            or "The selected checkpoint does not appear compatible with the selected config."
        )

    context = get_context(paths)
    upload_path = _save_uploaded_reference(uploaded_file)
    try:
        if upload_path:
            reference_waveform = load_waveform_file(
                upload_path,
                target_length=context.sample_length,
                target_sample_rate=context.sample_rate,
                target_channels=context.audio_channels,
            )
            reference_label = Path(uploaded_file.name).name
        else:
            reference_waveform, reference_label, _ = resolve_reference_waveform(
                context,
                class_name=reference_class,
                seed=seed,
                reference_split=reference_split,
                reference_index=reference_index,
            )

        class_id, conditioning = prepare_conditioning_inputs(context, target_class)
        model_input = build_noised_model_input(
            reference_waveform,
            model_device=context.model.device,
            seed=seed,
            noise_scale=noise_scale,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        timings = []
        generated = None
        with torch.inference_mode():
            for step_count in benchmark_steps:
                start = perf_counter()
                generated = generate_conditional_sample_from_input(
                    context,
                    model_input=model_input,
                    class_id=class_id,
                    conditioning_for_model=conditioning,
                    num_steps=step_count,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = perf_counter() - start
                duration = context.sample_length / context.sample_rate
                timings.append(
                    {
                        "steps": step_count,
                        "elapsed_ms": round(elapsed * 1000, 2),
                        "real_time_factor": round(elapsed / duration, 3),
                    }
                )

            if benchmark_steps[-1] != steps:
                start = perf_counter()
                generated = generate_conditional_sample_from_input(
                    context,
                    model_input=model_input,
                    class_id=class_id,
                    conditioning_for_model=conditioning,
                    num_steps=steps,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = perf_counter() - start
                timings.append(
                    {
                        "steps": steps,
                        "elapsed_ms": round(elapsed * 1000, 2),
                        "real_time_factor": round(elapsed / (context.sample_length / context.sample_rate), 3),
                    }
                )

        assert generated is not None
        reference_url = _write_audio(reference_waveform, context.sample_rate, "reference")
        model_input_audio = model_input.squeeze(0).detach().cpu()
        noisy_input_url = _write_audio(model_input_audio, context.sample_rate, "noisy-input")
        generated_url = _write_audio(generated, context.sample_rate, target_class)
        try:
            from realtime_demo.demo.metrics import build_reference_metrics

            reference_metrics = build_reference_metrics(
                clean=reference_waveform,
                noisy=model_input_audio,
                output=generated,
                sample_rate=context.sample_rate,
                output_dir=settings.MEDIA_ROOT,
                media_url=settings.MEDIA_URL,
            )
        except Exception as exc:
            reference_metrics = {
                "available": False,
                "message": f"Reference-based metrics could not be computed: {exc}",
            }
        peak_memory_mb = None
        if torch.cuda.is_available():
            peak_memory_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

        return {
            "reference_label": str(reference_label),
            "target_class": target_class,
            "sample_rate": context.sample_rate,
            "duration_sec": round(context.sample_length / context.sample_rate, 3),
            "reference_audio_url": reference_url,
            "noisy_input_audio_url": noisy_input_url,
            "generated_audio_url": generated_url,
            "benchmark": timings,
            "metrics": {
                "reference_rms": round(_rms(reference_waveform), 5),
                "noisy_input_rms": round(_rms(model_input_audio), 5),
                "generated_rms": round(_rms(generated), 5),
                "reference_centroid_hz": round(_spectral_centroid(reference_waveform, context.sample_rate), 2),
                "noisy_input_centroid_hz": round(_spectral_centroid(model_input_audio, context.sample_rate), 2),
                "generated_centroid_hz": round(_spectral_centroid(generated, context.sample_rate), 2),
                "peak_memory_mb": peak_memory_mb,
                "reference_metrics": reference_metrics,
            },
        }
    finally:
        if upload_path:
            Path(upload_path).unlink(missing_ok=True)
