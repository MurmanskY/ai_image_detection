# [CVPR Findings 2026] PRADA: Probability-Ratio-Based Attribution and Detection of Autoregressive-Generated Images

[![arXiv](https://img.shields.io/badge/arXiv-2511.20068-b31b1b)](https://arxiv.org/abs/2511.20068)

Simon Damm, Jonas Ricker, Henning Petzka, and Asja Fischer (Ruhr University Bochum)  
*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 2026*

This repository contains the official implementation of **PRADA** and information on how to use the **DARG dataset**.

## Setup
We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/).
Once installed, you can recreate our environment by running
```bash
uv sync
```
from within the root directory.
Depending on your environment, you might have to make changes to `pyproject.toml` to [select the right PyTorch version](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch).


## The DARG Dataset (**D**ataset of **A**uto**R**egressive-**G**enerated Images)

### Generated Images
Our dataset can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.20327169).
In total, It contains 155,000 images from 20 autoregressive image generators.
- Class-to-Image (10,000 images per model, 10 per [ImageNet](https://image-net.org) class)
  - [HMAR-d16/-d20/-d24/-d30](https://github.com/NVlabs/HMAR)
  - [LlamaGen-B/-L](https://github.com/foundationvision/llamagen)
  - [RAR-B/-L/-XL/-XXL](https://github.com/bytedance/1d-tokenizer/blob/main/README_RAR.md)
  - [VAR-d16/-d20/-d24/-d30/-d36](https://github.com/FoundationVision/VAR)
- Text-to-Image (1,000 images per model, prompts from [Synthbuster](https://zenodo.org/records/10066460), corresponding to real images from [RAISE-1k](https://loki.disi.unitn.it/RAISE))
  - [Infinity-2B](https://github.com/FoundationVision/Infinity)
  - [Janus-1B/-7B](https://github.com/deepseek-ai/Janus)
  - [LlamaGen-XL](https://github.com/foundationvision/llamagen)
  - [Switti-1024](https://github.com/yandex-research/switti)

**Note:** The evaluation in the paper uses a reduced subset.

**Generating your own images:** If you need more images or would like to reproduce our dataset, please refer to `scripts/generate_images.py`.

### Real Images
- Download the validation set from the [official ImageNet website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and run `uv run scripts/prepare_imagenet.py path/to/imagenet/val` to recreate out subset of 10000 images.
- Download `RAISE_1k.csv` from the [RAISE-1k website](https://loki.disi.unitn.it/RAISE/confirm.php?package=1k) and run `uv run scripts/prepare_raise1k.py path/to/RAISE_1k.csv` to download and convert the images.

## Reproducing Our Results
To reproduce our results for class-to-image models, run
```bash
uv run predict.py --real-image-dir data/darg/c2i/real --fake-image-dirs "['data/darg/c2i/hmar_d20','data/darg/c2i/hmar_d30','data/darg/c2i/llamagen_b256','data/darg/c2i/llamagen_l256','data/darg/c2i/rar_l','data/darg/c2i/rar_xxl','data/darg/c2i/var_d20','data/darg/c2i/var_d30']" --mode c2i --train-size 250 --output-root results/c2i
```

To reproduce our results for text-to-image models, run
```bash
uv run predict.py --real-image-dir data/darg/t2i/real --fake-image-dirs "['data/darg/t2i/infinity_2b','data/darg/t2i/janus_1b','data/darg/t2i/llamagen_xlstage2','data/darg/t2i/switti_1024']" --mode t2i --train-size 250 --output-root results/t2i
```

These commands will automatically use our pre-trained score functions.


## External Dependencies
Code in `src/external` contains external repositories (with slight modifications for fixing imports) that are subject to their own licenses:
- [HMAR](https://github.com/NVlabs/HMAR)
- [LlamaGen](https://github.com/foundationvision/llamagen)
- [VAR](https://github.com/FoundationVision/VAR)
- [RAR](https://github.com/bytedance/1d-tokenizer/blob/main/README_RAR.md)
- [Infinity](https://github.com/FoundationVision/Infinity)
- [Janus](https://github.com/deepseek-ai/Janus)
- [Switti](https://github.com/yandex-research/switti)

## Citation
If you find this repository or the DARG dataset helpful, please cite our work as follows:
```
@InProceedings{Damm_2026_CVPR,
    author    = {Damm, Simon and Ricker, Jonas and Petzka, Henning and Fischer, Asja},
    title     = {PRADA: Probability-Ratio-Based Attribution and Detection of Autoregressive-Generated Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {6506-6516}
}
```