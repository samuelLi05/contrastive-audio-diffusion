from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torchaudio

from main.inference_helpers import prepare_audio_for_display


EPS = 1e-10


def _mono_fixed(*samples: Any) -> list[torch.Tensor]:
    normalized = []
    min_len = None
    for sample in samples:
        audio = prepare_audio_for_display(sample).float().cpu()
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        elif audio.ndim > 2:
            audio = audio.reshape(-1)
        min_len = audio.numel() if min_len is None else min(min_len, audio.numel())
        normalized.append(audio)

    if min_len is None or min_len == 0:
        raise ValueError("Cannot compare empty audio tensors.")

    return [audio[:min_len] for audio in normalized]


def mse(clean: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(torch.mean((clean - candidate) ** 2).item())


def snr_db(clean: torch.Tensor, candidate: torch.Tensor) -> float:
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean((clean - candidate) ** 2) + EPS
    return float((10.0 * torch.log10((signal_power + EPS) / noise_power)).item())


def _stft(audio: torch.Tensor, *, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft)
    return torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        return_complex=True,
    )


def _power_spectrogram(audio: torch.Tensor, *, n_fft: int, hop_length: int) -> torch.Tensor:
    return _stft(audio, n_fft=n_fft, hop_length=hop_length).abs().square()


def _save_heatmap(
    image: torch.Tensor,
    *,
    output_dir: Path,
    media_url: str,
    stem: str,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    cmap: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{stem}-{uuid4().hex[:10]}.png"
    path = output_dir / filename

    fig = plt.figure(figsize=(8, 4.2))
    plt.imshow(image.numpy(), aspect="auto", origin="lower", cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=colorbar_label)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return f"{media_url}{filename}"


def _save_line_plot(
    x_values: list[float],
    y_values: list[float],
    *,
    output_dir: Path,
    media_url: str,
    stem: str,
    title: str,
    ylabel: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{stem}-{uuid4().hex[:10]}.png"
    path = output_dir / filename

    fig = plt.figure(figsize=(8, 3.4))
    plt.axhline(0.0, color="#88929a", linewidth=1)
    plt.plot(x_values, y_values, color="#0f766e", linewidth=2, marker="o", markersize=3)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return f"{media_url}{filename}"


def _worst_tf_regions(
    psnr_tf: torch.Tensor,
    *,
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    top_k: int,
) -> list[dict[str, float]]:
    flat_values, flat_indices = torch.topk((-psnr_tf).flatten(), k=min(top_k, psnr_tf.numel()))
    regions = []
    n_frames = psnr_tf.shape[1]
    for value, flat_idx in zip(flat_values, flat_indices):
        freq_bin = int(flat_idx.item() // n_frames)
        frame = int(flat_idx.item() % n_frames)
        regions.append(
            {
                "time_sec": round(frame * hop_length / sample_rate, 3),
                "freq_hz": round(freq_bin * sample_rate / n_fft, 1),
                "psnr_db": round(float(-value.item()), 2),
            }
        )
    return regions


def _segment_snr_improvement(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    output: torch.Tensor,
    *,
    sample_rate: int,
    window_sec: float = 0.125,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    window = max(1, int(sample_rate * window_sec))
    rows = []
    for start in range(0, clean.numel(), window):
        end = min(start + window, clean.numel())
        if end - start < max(16, window // 4):
            continue
        clean_seg = clean[start:end]
        noisy_seg = noisy[start:end]
        output_seg = output[start:end]
        noisy_snr = snr_db(clean_seg, noisy_seg)
        output_snr = snr_db(clean_seg, output_seg)
        rows.append(
            {
                "start_sec": round(start / sample_rate, 3),
                "end_sec": round(end / sample_rate, 3),
                "noisy_snr_db": round(noisy_snr, 2),
                "output_snr_db": round(output_snr, 2),
                "improvement_db": round(output_snr - noisy_snr, 2),
            }
        )
    worst = sorted(rows, key=lambda row: row["improvement_db"])[:5]
    return rows, worst


def build_reference_metrics(
    *,
    clean: Optional[Any],
    noisy: Optional[Any],
    output: Any,
    sample_rate: int,
    output_dir: Path,
    media_url: str,
) -> dict[str, Any]:
    if clean is None or noisy is None:
        return {
            "available": False,
            "message": "Reference-based metrics are disabled because clean/noisy reference audio is unavailable.",
        }

    clean_mono, noisy_mono, output_mono = _mono_fixed(clean, noisy, output)
    n_fft = min(1024, max(64, 2 ** int((min(clean_mono.numel(), 1024)).bit_length() - 1)))
    hop_length = max(1, n_fft // 4)

    clean_stft = _stft(clean_mono, n_fft=n_fft, hop_length=hop_length)
    output_stft = _stft(output_mono, n_fft=n_fft, hop_length=hop_length)
    clean_power = clean_stft.abs().square()
    output_power = output_stft.abs().square()
    noisy_power = _power_spectrogram(noisy_mono, n_fft=n_fft, hop_length=hop_length)

    error_tf = (clean_stft - output_stft).abs().square()
    max_power = clean_power.max().clamp_min(EPS)
    psnr_tf = 10.0 * torch.log10(max_power / (error_tf + EPS))
    psnr_tf = torch.clamp(psnr_tf, min=-20.0, max=80.0)

    mel = torchaudio.transforms.MelScale(
        n_mels=80,
        sample_rate=sample_rate,
        n_stft=clean_power.shape[0],
    )
    mel_diff = torch.log10((mel((noisy_power - output_power).abs()) + EPS))

    metrics_dir = output_dir / "metrics"
    metrics_url = f"{media_url}metrics/"
    psnr_url = _save_heatmap(
        psnr_tf,
        output_dir=metrics_dir,
        media_url=metrics_url,
        stem="tf-psnr",
        title="Time-Frequency PSNR vs Clean Reference",
        xlabel="Frame",
        ylabel="Frequency bin",
        colorbar_label="PSNR (dB)",
        cmap="magma",
    )
    mel_diff_url = _save_heatmap(
        mel_diff,
        output_dir=metrics_dir,
        media_url=metrics_url,
        stem="mel-before-after-diff",
        title="Before/After Mel Power Difference",
        xlabel="Frame",
        ylabel="Mel bin",
        colorbar_label="log10 |noisy power - output power|",
        cmap="viridis",
    )

    segments, worst_segments = _segment_snr_improvement(
        clean_mono,
        noisy_mono,
        output_mono,
        sample_rate=sample_rate,
    )
    segment_url = None
    if segments:
        segment_url = _save_line_plot(
            [row["start_sec"] for row in segments],
            [row["improvement_db"] for row in segments],
            output_dir=metrics_dir,
            media_url=metrics_url,
            stem="segment-snr-improvement",
            title="Segment-wise SNR Improvement",
            ylabel="Output SNR - Noisy SNR (dB)",
        )

    return {
        "available": True,
        "message": "",
        "mse_noisy": round(mse(clean_mono, noisy_mono), 8),
        "mse_output": round(mse(clean_mono, output_mono), 8),
        "snr_noisy_db": round(snr_db(clean_mono, noisy_mono), 2),
        "snr_output_db": round(snr_db(clean_mono, output_mono), 2),
        "snr_improvement_db": round(snr_db(clean_mono, output_mono) - snr_db(clean_mono, noisy_mono), 2),
        "psnr_heatmap_url": psnr_url,
        "mel_difference_url": mel_diff_url,
        "segment_snr_url": segment_url,
        "worst_tf_regions": _worst_tf_regions(
            psnr_tf,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            top_k=5,
        ),
        "worst_segments": worst_segments,
    }
