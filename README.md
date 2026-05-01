# Contrastive Audio Diffusion

Authors: Samuel Li, Zachary Houlton, Ariv Mondal, Daniel Zhu

Originally forked from an earlier tiny-audio-diffusion repository; this codebase has been refactored and extended into a new, standalone project focused on conditional waveform diffusion with contrastive label alignment. The original project is acknowledged here once as a starting point. 
https://github.com/crlandsc/tiny-audio-diffusion 

## Motivation

Contrastive Audio Diffusion provides a focused codebase for training and running conditional waveform diffusion models (1‑D U‑Net) that use contrastive learning to align label embeddings with audio embeddings. The aim is to make conditional waveform diffusion experiments reproducible and extensible, especially for NSynth‑based conditional workflows.


## Background

Direct waveform diffusion is inherently computationally intensive. For example, an audio sample with the industry standard 44.1kHz sampling rate requires 44,100 samples for just 1 second of audio. Now multiply that by 2 for a stereo file. However, it has a significant advantage over many methods that reduce audio into spectrograms or downsample - the network retains and learns from *phase* information. Phase is challenging to represent on its own in visual methods, such as spectrograms, as it appears similar to that of random noise. Because of this, many generative methods discard phase information and then implement ways of estimating and regenerating it. However, it plays a key role in defining the timbral qualities of sounds and should not be dispensed with so easily.

Waveform diffusion is able to retain this important feature as it does not perform any transforms on the audio before feeding it into the network. This is how humans perceive sounds, with both amplitude and phase information bundled together in a single signal. As mentioned previously, this comes at the expense of computational requirements and is often reserved for training on a cluster of GPUs with high speeds and lots of memory. Because of this, it is hard to begin to experiment with waveform diffusion with limited resources.

This repository seeks to offer some base code to those looking to experiment with and learn more about waveform diffusion on their own computer without having to purchase cloud resources or upgrade hardware. This goes for not only *inference*, but *training* your own models as well!

To make this feasible, however, there must be a tradeoff of quality, speed, and sample length. Because of this, we have focused on training base models for one-shot drum samples - as they are inherently short in sample length.

The current configuration is set up to be able to train ~0.75 second stereo samples at 44.1kHz, allowing for the generation of high-quality one-shot audio samples. The network configuration can be adjusted to improve the resolution, sample rate, training and inference speed, sample length, etc. but, of course, more hardware resources will be required.

---

## Setup

<div align="center">
  <h1 style="font-size: 36px;">Contrastive Audio Diffusion</h1>
</div>

Built from an initial tiny‑audio‑diffusion codebase; this repository has been refocused and extended to provide conditional waveform diffusion with contrastive label alignment. The project name is now "Contrastive Audio Diffusion" and the code, configs, and documentation below reflect this conditional mode as the primary workflow.

**Overview**

Contrastive Audio Diffusion trains waveform diffusion models that generate short audio samples (seconds or sub‑second clips) conditioned on instrument labels. The key components are:

- A 1‑D U‑Net denoiser wrapped by a diffusion sampler (VDiffusion).
- Optional attention and cross‑attention inside the U‑Net for long‑range context and conditioning fusion.
- Two conditioning modes: one‑hot concatenation and learnable label embedding (with optional cross‑attention).
- An auxiliary contrastive loss that aligns label embeddings with audio embeddings so conditioning vectors become semantically meaningful.

Recommended background reading: Audio denoising and diffusion concepts are well explained in this Medium post: https://medium.com/@zacharyhoulton/audio-denoising-using-diffusion-c2ae04d20c4e

**Why conditional + contrastive?**

The diffusion denoiser learns to reconstruct clean waveforms from noisy inputs. Contrastive learning forces label embeddings to be close to their matching clean audio embeddings and far from mismatched ones. When used together, the model learns not only how to denoise, but how to denoise with intent: the conditioning vector becomes an "instruction" the U‑Net can follow to produce the desired instrument timbre.

---

**Quick Start**

1. Create the environment and activate it (see `setup/environment.yml` and `setup/requirements.txt`):

```bash
conda env create -f setup/environment.yml
conda activate contrastive-audio-diffusion
```

2. Install the Jupyter kernel for this environment (optional but recommended for the notebook):

```bash
python -m ipykernel install --user --name contrastive-audio-diffusion --display-name "contrastive-audio-diffusion"
```

3. Edit `.env` (rename from `.env.tmp`) to set `DIR_DATA`, `DIR_LOGS`, and optional W&B variables if you want logging.

---

**Datasets**

This repo uses processed NSynth metadata and waveform exports for conditional training. The conditional datamodule expects a `metadata.jsonl` file that lists each sample's `Clean Path`, optional `Noisy Path`, `class` label, and any precomputed conditioning vectors. The included helper scripts and the datamodule make training reproducible from this metadata format.

If you want to prepare a small NSynth subset locally, use `scripts/prepare_nsynth_subset.py`. For processed metadata exports, point `datamodule.metadata_path` in your experiment config to the `metadata.jsonl` file.

---

**Training (conditional)**

This repository includes convenience scripts and Hydra experiments for conditional training. Key points:

- `scripts/train_conditional_models.sh` — wrapper to launch conditional experiments (onehot, embedding, or both). Use this to quickly run the common variants.
- Hydra experiments in `exp/` provide preconfigured model sizes and training settings. See `exp/nsynth_conditional_16gb_embedding_no_wandb.yaml` for an example embedding+contrastive setup.

Examples:

```bash
# One‑hot conditioning (no contrastive loss)
scripts/train_conditional_models.sh onehot data/nsynth_waveform_processed/metadata/metadata.jsonl

# Label embedding + contrastive loss
scripts/train_conditional_models.sh embedding data/nsynth_waveform_processed/metadata/metadata.jsonl
```

Or call `train.py` directly with Hydra overrides:

```bash
PYTHONPATH=. python train.py exp=nsynth_conditional_16gb_embedding_no_wandb datamodule.metadata_path=data/nsynth_waveform_processed/metadata/metadata.jsonl model.conditioning_mode=label_embedding model.use_contrastive_loss=true
```

Notes:
- During training the pipeline uses the clean waveform as the ground truth for the denoising objective and also as the audio anchor for the contrastive objective. See `main/diffusion_module.py` for the `training_step` implementation.
- The code supports training with clean‑only datasets (no noisy inputs) — the dataset loader gracefully handles missing noisy paths and falls back to clean waveforms.

---

**Inference**

Use the interactive `Inference.ipynb` to load a checkpoint and generate samples. The notebook supports:

- Loading metadata to auto‑derive class names and conditioning dims.
- One‑shot unconditional and class‑conditioned generation.
- Reference‑based conditional generation (style transfer from a reference waveform).

Set `metadata_path_override` in the notebook to the metadata used during training to ensure class names and conditioning dims match the checkpoint.

**Django Real‑Time Demo**

This repository includes a lightweight Django demo that exercises the inference path for near‑real‑time conditional audio morphing and benchmarking. The demo reuses the `main/inference_helpers.py` utilities, caches a loaded model instance, runs generation in inference mode, and measures latency / real‑time factor for multiple sampler step counts.

To run the demo, set the model, checkpoint and metadata paths (or enter them in the web UI form) and start the server. On Linux/macOS use environment exports, for example:

```bash
export TAD_CONFIG_PATH=exp/nsynth_conditional_16gb_embedding_no_wandb.yaml
export TAD_CKPT_PATH=logs/ckpts/<run-folder>/<checkpoint>.ckpt
export TAD_METADATA_PATH=data/nsynth_waveform_processed/metadata/metadata.jsonl
export TAD_CONDITIONING_MODE=label_embedding
export TAD_CLASS_NAMES=bass,brass,flute,guitar,keyboard,mallet,organ
python manage.py runserver 127.0.0.1:8000
```

Open `http://127.0.0.1:8000/` in your browser. The UI lets you:

- upload a short reference audio file or pick a dataset reference sample,
- select a target class from the configured class list,
- trigger conditional generation and view output audio, and
- inspect latency and real‑time‑factor graphs for several sampler step counts.

If you prefer to avoid restarting the server when changing models, paste the desired checkpoint, config, metadata path, and conditioning mode directly into the demo form — the server will load the new model dynamically.

If no trained conditional checkpoint is available locally, you may use a compatible, untrained checkpoint for UI and latency benchmarking only. It will load the correct architecture and exercise the generation path, but it will not produce meaningful audio quality.

Place the Django app in the `web/` folder (if present) or consult `web/README.md` for deployment notes and dependency installation.

**Model architecture**

High level:

- `UNetV0` (1‑D U‑Net) is the denoiser core used by `DiffusionModel` and `VDiffusion`.
- Attention and cross‑attention are optionally enabled per U‑Net stage to improve long‑range context and conditioning fusion.
- `ConditionalModel` implements concatenative conditioning (one‑hot) and `EmbeddingConditionalModel` implements embedding/cross‑attention conditioning with optional contrastive heads (`audio_to_latent`, `text_embedding`, `text_to_latent`).

If you want architectural intuition, this Medium post covers diffusion and denoising concepts in approachable detail: https://medium.com/@zacharyhoulton/audio-denoising-using-diffusion-c2ae04d20c4e

---

**Where to look in the code**

- `main/diffusion_module.py` — model wrappers, conditional training_step, contrastive loss, datamodule.
- `main/inference_helpers.py` — checkpoint loading, metadata parsing, and inference utilities.
- `main/nsynth_waveform_dataset.py` — metadata format, waveform loading, and dataset handling.
- `exp/` — experiment YAML files for Hydra. Edit these for different U‑Net sizes, conditioning modes, and sampling settings.
- `scripts/train_conditional_models.sh` — convenience launcher for conditional experiment runs.
- `Inference.ipynb` — interactive example for loading checkpoints, sampling, and diagnostics.

---

**License & attribution**

This project is released under the MIT license. Maintainers: Samuel Li, Zachary Houlton, Ariv Mondal, Daniel Zhu. The current repository is the canonical source for "Contrastive Audio Diffusion" experiments, datasets, and training recipes.


