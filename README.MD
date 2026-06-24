<div align="center">
  <h1>Modality-Inconsistent Continual Learning of Multimodal Large Language Models</h1>
  <p>
    Transactions on Machine Learning Research (TMLR), 2026
  </p>
  <p>
    <a href="https://openreview.net/pdf?id=FD8or43fBU"><img src="https://img.shields.io/badge/Paper-OpenReview-8A2BE2" alt="Paper"></a>
    <a href="#data-preparation"><img src="https://img.shields.io/badge/Data-preparation-1F6FEB" alt="Data preparation"></a>
    <a href="#running"><img src="https://img.shields.io/badge/Code-training-181717" alt="Training code"></a>
  </p>
</div>

<!-- This repository runs multimodal continual learning experiments over six tasks. Two task orders are provided:

- [`run_order_1.sh`](run_order_1.sh)
- [`run_order_2.sh`](run_order_2.sh)

Each script contains one commented command block per training step. The blocks are intentionally disabled so that a run executes exactly one step at a time and uses the checkpoint produced by the previous step. -->

<!-- ## Before running

Run commands from the repository root. Configure the paths to the datasets, pretrained models, and any required checkpoints for your environment before launching training. The default experiment configuration is [`config_llama_3.2_only_proj.yaml`](config_llama_3.2_only_proj.yaml); its `llm_model` field defaults to `meta-llama/Llama-3.2-1B-Instruct`.

The launcher commands use `CUDA_VISIBLE_DEVICES=0` and `nohup`. Change the GPU index, remove `nohup`, or adjust log-file paths as needed. -->

# Data Preparation

Download the original data from the respective dataset providers and place the media files under `dataset/` using the following layout. The file names must match the IDs in the processed annotations; do not rename the downloaded media files.

```text
dataset/
├── flickr30k/
│   └── flickr30k-images/
│       └── <image_id>.jpg
├── coco/
│   └── OK-VQA/
│       ├── train2014/
│       │   └── COCO_train2014_<12-digit-image-id>.jpg
│       └── val2014/
│           └── COCO_val2014_<12-digit-image-id>.jpg
├── MSRVTT/
│   └── videos/
│       └── all/
│           └── <video_id>.mp4
├── AudioCaps/
│   └── all_audios/
│       └── <audio_id>.wav
├── Clotho-AQA/
│   └── audio_files/
│       └── <audio_id>
└── MSVD/
    └── MSVD-QA/
        └── video/
            └── <video_id>
```

The six datasets used in this work are Flickr30k, OK-VQA, MSR-VTT, AudioCaps, Clotho-AQA, and MSVD-QA. Their required media locations and associated tasks are:

| Dataset | Task | Required media directory |
| --- | --- | --- |
| Flickr30k | Image captioning | `dataset/flickr30k/flickr30k-images/` |
| OK-VQA | Image question answering | `dataset/coco/OK-VQA/{train2014,val2014}/` |
| MSR-VTT | Video captioning | `dataset/MSRVTT/videos/all/` |
| AudioCaps | Audio captioning | `dataset/AudioCaps/all_audios/` |
| Clotho-AQA | Audio question answering | `dataset/Clotho-AQA/audio_files/` |
| MSVD-QA | Video question answering | `dataset/MSVD/MSVD-QA/video/` |

Processed annotations are loaded from `data/`. Ensure that the following files are present:

```text
data/
├── all_img_id_caption_dict.npy
├── all_vid_caption_dict.npy
├── all_audio_id_cap_dict.npy
├── question_id_img_id_quesion_answer_dict.npy
├── question_id_video_id_quesion_answer_dict.npy
├── question_id_audio_id_quesion_answer_dict.npy
└── Natural_Instructions/
    └── data.json
```

The final file, `data/Natural_Instructions/data.json`, is used for the text-instruction constraint during training. The pre-generated pseudo targets used by launcher blocks with `--pseudo` are read from `pseudo_targets/` and should remain at their existing paths.

# Environment

We conduct experiments with Python 3.9.18, PyTorch 2.2.1, and CUDA 12.1.

To set up the environment, please simply run:

```bash
pip install -r requirements.txt
```

# Training

For either order, steps must be run **sequentially**: `0`, `1`, `2`, `3`, `4`, then `5`.

For a selected step:

1. Open the corresponding `run_order_*.sh` file.
2. Find the block labelled `Uncomment the following script for training step <N>`.
3. Remove the leading `#` from **every line of that block only**.
4. Leave every other step block commented out.
5. Launch the script with `sh run_order_<ORDER>.sh`.
6. Wait for the job to finish and confirm that its checkpoint has been written before continuing to the next step.
7. Re-comment the completed block, uncomment the next block, and repeat.

Steps after step 0 load the best checkpoint from the preceding step.

## Order 1

Order 1 follows this task sequence:

| Step | Task | Dataset | Script block |
| ---: | --- | --- | --- |
| 0 | Audio Captioning | AudioCaps | `run_order_1.sh` — step 0 |
| 1 | Image Captioning | Flickr30k | `run_order_1.sh` — step 1 |
| 2 | Video Question-Answering | MSVD-QA | `run_order_1.sh` — step 2 |
| 3 | Audio Question-Answering | ClothoAQA | `run_order_1.sh` — step 3 |
| 4 | Image Question-Answering | OK-VQA | `run_order_1.sh` — step 4 |
| 5 | Video Captioning | MSR-VTT | `run_order_1.sh` — step 5 |

To start order 1, uncomment only its step-0 block and run:

```bash
sh run_order_1.sh
```

After it completes, repeat the same process with the step-1 block, continuing through step 5.

## Order 2

Order 2 uses the same six tasks in a different sequence:

| Step | Task | Dataset | Script block |
| ---: | --- | --- | --- |
| 0 | Image Captioning | Flickr30k | `run_order_2.sh` — step 0 |
| 1 | Video Captioning | MSR-VTT | `run_order_2.sh` — step 1 |
| 2 | Video Question-Answering | MSVD-QA | `run_order_2.sh` — step 2 |
| 3 | Image Question-Answering | OK-VQA | `run_order_2.sh` — step 3 |
| 4 | Audio Captioning | AudioCaps | `run_order_2.sh` — step 4 |
| 5 | Audio Question-Answering | ClothoAQA | `run_order_2.sh` — step 5 |

To start order 2, uncomment only its step-0 block and run:

```bash
sh run_order_2.sh
```

Then progress through steps 1--5 in order, using the same uncomment-run-wait procedure.

# Citation

If you find this work useful, please consider citing it.
```bibtex
@article{pian2026modalityinconsistent,
  title={Modality-Inconsistent Continual Learning of Multimodal Large Language Models},
  author={Weiguo Pian and Shijian Deng and Shentong Mo and Mingrui Liu and Yunhui Guo and Yapeng Tian},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026}
}
```
