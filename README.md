# Voice Anti-Spoofing: LCNN on ASVspoof2019 LA

LCNN-based countermeasure for detecting spoofed speech (TTS/voice conversion), trained on the LA
partition of [ASVspoof2019](https://www.asvspoof.org/index2019.html).

**Result:** eval EER = 6.90%

## Project structure

- `src/model/` — LCNN and MFM.
- `src/transforms/spectrogram.py` — convertion to spectrogram
- `src/datasets/asvspoof_dataset.py` — parses ASVspoof2019 LA protocols, load flac.
- `src/trainer/trainer.py` — training loop, computes EER on eval each epoch.
- `src/trainer/inferencer.py` — inference on eval set
- `src/configs/asvspoof.yaml` / `asvspoof_inference.yaml` — training / inference configs.

## Installation

```bash
pip install -r requirements.txt
pre-commit install
```

## Usage

Train:

```bash
python3 train.py --config-name=asvspoof writer.run_name=<run_name> \
    dataloader.batch_size=<batch_size> lr_scheduler.T_max=<total_optimizer_steps>
```

`T_max` = `epoch_len(batch_size) × n_epochs` (recompute if either changes).

Run inference on eval / generate submission CSV:

```bash
python3 inference.py --config-name=asvspoof_inference \
    inferencer.from_pretrained=saved/<run_name>/model_best.pth \
    inferencer.submission_name=<your_university_username>
```

## WandB Report

tbd
