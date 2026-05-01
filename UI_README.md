### Django Real-Time Demo

This repository also includes a lightweight Django demo for near-real-time conditional audio morphing and integrated generation benchmarking. It reuses the existing inference helpers, caches the loaded model, wraps generation in inference mode, and benchmarks multiple sampler step counts without triggering any training.

Set the checkpoint/config paths for the model you want to demo or change it in the UI form:

```bash
set TAD_CONFIG_PATH=exp/nsynth_conditional_16gb_embedding_no_wandb.yaml
set TAD_CKPT_PATH=logs/ckpts/<run-folder>/<checkpoint>.ckpt
set TAD_METADATA_PATH=data/nsynth_waveform_processed/metadata/metadata.jsonl
set TAD_CONDITIONING_MODE=label_embedding
set TAD_CLASS_NAMES=bass,brass,flute,guitar,keyboard,mallet,organ
python manage.py runserver 127.0.0.1:8000
```

Then open `http://127.0.0.1:8000/`. You can upload a short reference audio file, choose a target class from the configured class list, generate a transformed sample, and view the latency/real-time-factor graph for several sampler step counts.

If you do not want to restart the server when changing models, paste the checkpoint, config, metadata, and conditioning-mode values directly into the demo form. Uploaded reference audio does not require metadata; dataset reference sampling does.

If no trained conditional checkpoint is available locally, a compatible untrained checkpoint can be used for UI and latency benchmarking only. It will load the correct architecture and exercise the real generation path, but it will not produce meaningful audio quality.

