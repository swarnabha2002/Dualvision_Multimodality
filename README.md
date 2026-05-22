# DualVision: RGB–Infrared Multimodal Large Language Models for Robust Visual Reasoning

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2604.18829)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://abrarmajeedi.github.io/dualvision)
[![CVPR 2026](https://img.shields.io/badge/CVPR%20Findings-2026-1f77b4.svg)](https://abrarmajeedi.github.io/dualvision)

Official repository for our **CVPR Findings 2026** paper:
**"DualVision: RGB–Infrared Multimodal Large Language Models for Robust Visual Reasoning"**

[Abrar Majeedi](https://abrarmajeedi.github.io)<sup>1</sup>,
Zhiyuan Ruan<sup>2</sup>,
Ziyi Zhao<sup>2</sup>,
Hongcheng Wang<sup>2</sup>,
Jianglin Lu<sup>3</sup>,
Yin Li<sup>1</sup>

<sup>1</sup>University of Wisconsin–Madison &nbsp;&nbsp; <sup>2</sup>Amazon &nbsp;&nbsp; <sup>3</sup>Northeastern University

---

## 🚧 Code & Data Coming Soon

> ### ⏳ **Code and dataset annotations will be released in a few weeks. Stay tuned!** ⏳
In the meantime, please visit the [project page](https://abrarmajeedi.github.io/dualvision) for full details, qualitative examples, and quantitative results.

---

## Overview

Multimodal large language models (MLLMs) achieve impressive performance on visual reasoning with RGB imagery, yet remain fragile under common degradations such as fog, blur, and low-light conditions. Infrared (IR) imaging is a long-established complement to RGB and offers inherent robustness in these settings, but its integration into MLLMs is underexplored.

**DualVision** is a lightweight fusion module that efficiently incorporates IR–RGB information into MLLMs via patch-level **multi-scale localized cross-attention**. Each RGB patch token attends only to spatially corresponding IR regions, injecting complementary IR cues where they are most relevant — with low compute overhead and broad compatibility with existing MLLMs.

<p align="center">
  <img src="https://abrarmajeedi.github.io/dualvision/img/highlevel_cropped.png" width="80%">
</p>

## Key Contributions

- **DualVision fusion module** — a lightweight IR–RGB fusion design built on multi-scale 2D local cross-attention, drop-in compatible with existing MLLMs.
- **DV-204K** — a dataset of ~25K aligned IR–RGB image pairs with ~204K modality-aware QA annotations for instruction tuning.
- **DV-500** — a curated benchmark of 500 IR–RGB image pairs with 500 QA pairs for evaluating cross-modal reasoning under degradations.
- **~75% compute savings** vs. naïve concatenation: RGB-IR concatenation costs O(4N²); DualVision fuses to N tokens at O(N²).
- **Strong empirical results** under blur, low-light, and fog — outperforming open- and closed-source MLLMs (LLaVA 1.5, Qwen2-VL, LLaVA-Next Interleave, LLaMA-4 Scout, Claude Sonnet 3.5v2, Claude Opus 4) across clean and degraded settings.

## Method

<p align="center">
  <img src="https://abrarmajeedi.github.io/dualvision/img/main_fig.png" width="100%">
</p>

- **2D Local Cross-Attention** — each RGB token attends to IR tokens within a radius-r region centered at its location, enforcing spatially aligned fusion.
- **Multi-Scale Design** — multiple local cross-attention blocks applied sequentially with progressively larger radii, capturing multi-scale interactions while preserving locality.
- **Computational Efficiency** — fuses both modalities into N tokens (4× fewer than concatenation), reducing attention cost by ~75%.

## Datasets

We build **DV-204K** and **DV-500** on top of publicly available aligned IR–RGB image pairs. The underlying images can be downloaded from their official sources:

- [LLVIP](https://bupt-ai-cz.github.io/LLVIP/)
- [HDRT](https://www.sciencedirect.com/science/article/abs/pii/S1566253525001824)

The QA annotations for DV-204K and DV-500 will be released in a few weeks!

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{majeedi2026dualvision,
  title={DualVision: RGB-Infrared Multimodal Large Language Models for Robust Visual Reasoning},
  author={Abrar Majeedi and Zhiyuan Ruan and Ziyi Zhao and Hongcheng Wang and Jianglin Lu and Yin Li},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year={2026}
}
```

## Contact

For questions, please email majeedi@wisc.edu
