# Seer: Language Instructed Video Prediction with Latent Diffusion Models

This repository is the official PyTorch implementation for Seer introduced in the paper:

[Seer: Language Instructed Video Prediction with Latent Diffusion Models](https://arxiv.org/abs/2303.14897). ICLR 2024
<br>
Xianfan Gu, Chuan Wen, Weirui Ye, Jiaming Song, and Yang Gao
<br>
## Approach

https://github.com/seervideodiffusion/SeerVideoLDM/assets/12033780/9486670d-4945-4533-aae5-8a6dbf6b5248



## Installation

### Dependency Setup
* Python 3.8
* PyTorch 1.12.0
* Other dependencies

Create an new conda environment.
```
conda create -n seer python=3.8 -y
conda activate seer
```
Install following packages and environments in the following (Note: only accelerate, transformers, xformers, diffusers require indicated versions):
```
pip install -r requirements.txt
```

### Dataset Preparation 
The deault datasets include [Something-Something v. 2 (Sthv2)](https://developer.qualcomm.com/software/ai-datasets/something-something), [Bridge Data](https://sites.google.com/view/bridgedata), and [Epic Kitchens](https://epic-kitchens.github.io/2023) to fine-tune and evaluate Seer model. See [MMAction2](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/sthv2/README.md) for detailed steps about video frames extraction.

The overall file structure in Sthv2:
```
data
├── annotations
│   ├── train.json
|   ├── validation.json
|   └── test.json
├── rawframes
│   ├── 1
|   |   ├── img_00001.jpg
|   |   ├── img_00002.jpg
|   |   ├── img_00003.jpg
|   |   └── ...
│   ├── 2
│   └── ...
├── videos
│   ├── 1.mp4
│   ├── 2.mp4
│   └── ...
```
The overall file structure in Bridge Data (name of subfolder follow path list in [dataset/path_id_bridgedata.txt](dataset/path_id_bridgedata.txt)):
```
data
├── rawframes
│   ├── close_brown1fbox_flap
|   |   ├── 0
|   |   |   ├── img_00001.jpg
|   |   |   ├── img_00002.jpg
|   |   |   ├── img_00003.jpg
|   |   |   └── ...
|   |   ├── 1
|   |   ├── 2
|   |   └── ...
│   ├── close_small4fbox_flaps
│   └── ...
```
The overall file structure in Epic Kitchens:
```
data
├── P01
│   └── rgb_frames
|       ├── P01_01
|       |   ├── frame_0000000001.jpg
|       |   ├── frame_0000000002.jpg
|       |   └── ...
|       ├── P01_02
|       └── ...
├── P02
└── ...
```

## Fine-tune Seer From Inflated Stable Diffusion
1.Download the initialization checkpoint of FSText model ([download](https://huggingface.co/xianfang/SeerVideo/tree/main/fstext_init)), then place it under `store_pth/fstext_init` 

2.To fine-tune with 24GB NVIDIA 3090 GPUs by running:
```
accelerate launch train.py --config ./configs/train.yaml
``` 
The default version of Stable-Diffusion is `runwayml/stable-diffusion-v1-5`.

## Inference

### Checkpoint of Seer
The checkpoints fine-tuned on Dataset Something-Something v. 2 (Sthv2), Bridge Data, and Epic Kitchens can be downloaded as following:

| Dataset | training steps| num ref.frames | num frames |Link | 
| :---: | :---: | :---:  | :---: | :---: |
| Sthv2 | 200k| 2 | 12 |  [[checkpoints](https://huggingface.co/xianfang/SeerVideo/tree/main/sthv2_seer)]   |
| Bridge Data | 80k| 1 | 16 |  [[checkpoints](https://huggingface.co/xianfang/SeerVideo/tree/main/bridge_seer)]   |
| Epic Kitchens | 80k| 1 | 16 |  [[checkpoints](https://huggingface.co/xianfang/SeerVideo/tree/main/epickitchen_seer)]   |
| Sthv2+Bridge | 200k+80k| 1 | 16 |  [[checkpoints](https://huggingface.co/xianfang/SeerVideo/tree/main/sthv2bridge_seer)]   |

After downloading a checkpoint file, place it under `outputs/` folder and set `output_dir` attributes in `inference.yaml` or `eval.yaml`.

### Inference of Dataset
The inferece stage of Seer. To sample batches of video clip and visualize results from dataset by running (Ensure that the checkpoint file is existed)
```
accelerate launch inference.py --config ./configs/inference.yaml
``` 
### Sampling Video Clip from Image
To sample a video clip from an indicated image:
```
python inference_img.py \
 --config="./configs/inference_base.yaml" \
 --image_path="./src/figs/{image_name}.jpg" \
 --input_text_prompts="{your input text}"
```
For Example
```
python inference_img.py\
--config="./configs/inference_base.yaml"\ --image_path="./src/figs/book.jpg"\ --input_text_prompts="close book"
```
(Hints: We recommend using Sthv2+Bridge [checkpoints](https://huggingface.co/xianfang/SeerVideo/tree/main/sthv2bridge_seer) for improved performance in zero-shot video prediction tasks.)

## Evaluation

### FVD/KVD Metrics
The evaluation of FVD/KVD is based on the implementation of [VideoGPT](https://github.com/wilson1yan/VideoGPT). See the evaluation results by setting `compute_fvd:True` in `eval.yaml` and running:
```
accelerate launch eval.py --config ./configs/eval.yaml
``` 

## Citation

If you find this repository useful, please consider giving a star :star: and citation:
```latex
@article{gu2023seer,
    author  = {Gu, Xianfan and Wen, Chuan and Ye, Weirui and Song, Jiaming and Gao, Yang},
    title   = {Seer: Language Instructed Video Prediction with Latent Diffusion Models},
    journal = {arXiv preprint arXiv:2303.14897},
    year    = {2023},
}
```

## Acknowledgement
This code builds on [Diffusers](https://github.com/huggingface/diffusers) and modified from [Tune-A-Video](https://github.com/showlab/Tune-A-Video).
