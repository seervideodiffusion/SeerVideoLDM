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
1.Download the initialization checkpoint of FSText model ([download](https://cdn-lfs-us-1.huggingface.co/repos/a2/2c/a22cdbf99427d532d5a6fb63746b474b12a944684e9f0781c21e402b21362d7f/e9cef0a98815790306f315f05b71a1360604047707f5bcffb2c5778fd595ab15?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27pytorch_model.bin%3B+filename%3D%22pytorch_model.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1708744261&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcwODc0NDI2MX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zL2EyLzJjL2EyMmNkYmY5OTQyN2Q1MzJkNWE2ZmI2Mzc0NmI0NzRiMTJhOTQ0Njg0ZTlmMDc4MWMyMWU0MDJiMjEzNjJkN2YvZTljZWYwYTk4ODE1NzkwMzA2ZjMxNWYwNWI3MWExMzYwNjA0MDQ3NzA3ZjViY2ZmYjJjNTc3OGZkNTk1YWIxNT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=l2U6eMXiJYMunYh4ayPTDsTrL1bWps7lN0dTHWtxC6krHxRfvZPWwYiykJ-kYLsFikL5ywmDL2ctNkjd9nl0vKo8WEBh09-LQrw30WPIhhukbY7DEKxbEI1wK-NtgOrNbEPGYuYyiXXFOQCZRJ01Lc3bfkwL7elgMvsOJiISzW9g25RkpA-r6zrfW5XnsZHuJaLsPF1VRpZzlReZitu50b1TmSvLMXgbUZvIhs4DK1OzN6AJIzYGk4x0xPUH%7E9nFS2tn389BNzYc1GscHKvoEeFJ2lIoiLqGtDuHqNPwX4m3mzj%7EJYQJqKIho1OQV4Dl5ewkXpxkC6NnghNGG3kOBQ__&Key-Pair-Id=KCD77M1F0VK2B)), then place it under `store_pth/fstext_init` 

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
