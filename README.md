# Dynamic Edge Pipeline for MLLM IoT Video Analysis

This repository contains the codebase for the MTP thesis: **"SEC: Enabling MLLMs for Low-Latency IoT Video Analysis via Semantic-Aware Edge-Cloud Collaboration"**.

## Core Contributions (Implemented)

### Contribution 1: Query-Aware Dynamic Temporal Segmentation
Instead of a fixed segmentation approach, the edge device is equipped with a lightweight NLP query analyzer (`query_analyzer.py`). Before processing the video, the AI evaluates the text query to predict the required **Temporal Granularity**. Fast-action queries scale the chunk size up, while static queries scale it down to save battery.

### Contribution 2: Audio-Triggered Asynchronous Frame Extraction
To complement the Query-Aware segmentation, we introduce an asynchronous audio-visual listener (`edge_frame_selection.py`). If a sudden audio spike or specific sound class is detected, it acts as an **interrupt trigger**, forcing the edge device to instantly extract candidate frames at the exact timestamp of the sound.

---

## Future Work

### Iterative Cloud-to-Edge Feedback Loop
We propose transforming the standard one-way (Edge -> Cloud) pipeline into an active, agentic retrieval system. The edge device maintains a rolling buffer of high-resolution video. If the massive Cloud MLLM begins drafting its answer but drops in confidence, it sends a tiny **Fetch Request** back to the edge (e.g., *"Send me 10 continuous frames spanning exactly from timestamp 00:15 to 00:16."*)

## Evaluation
To run the ablation study comparing our Dynamic Edge method against Uniform Sampling:
```bash
cd dynamic_edge
python paper_evaluator.py
```

Alternatively, upload `colab_academic_evaluator.ipynb` to Google Colab to evaluate against the UCF101 dataset on a cloud GPU.
