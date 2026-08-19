---
title: NeuroTRIBE free visual inference
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Visual-only TRIBE v2 inference for the local NeuroTRIBE demo
tags:
  - zerogpu
  - neuroscience
  - fmri
---

# NeuroTRIBE free ZeroGPU inference

This public Hugging Face Space receives only a short stimulus video and returns
TRIBE v2's mean visual cortical prediction as a `.npy` file. It is designed for
a few interactive demo runs, not persistent or clinical service.

Observed BOLD fMRI is never uploaded: the local NeuroTRIBE API keeps the BOLD
projection, deviation calculation, maps, and report on the user's computer.

The Space has no API secrets and is deliberately public so it can use the free
ZeroGPU tier. Do not submit private stimulus media.
