QPMIL-VL ✨ <a href="https://www.scse.uestc.edu.cn/"> <img width="20%" src="./docs/school.png" align="right" /> </a>
===========

## Queryable Prototype Multiple Instance Learning with Vision-Language Models for Incremental Whole Slide Image Classification

Accepted by *[AAAI 2025](https://aaai.org/conference/aaai/aaai-25/)*

[[Paper]](https://arxiv.org/abs/2410.10573) | [[Running the Code]](https://github.com/can-can-ya/QPMIL-VL?tab=readme-ov-file#-running-the-code) | [[Acknowledgements]](https://github.com/can-can-ya/QPMIL-VL?tab=readme-ov-file#-acknowledgements) | [[License and Terms of Use]](https://github.com/can-can-ya/QPMIL-VL?tab=readme-ov-file#-license-and-terms-of-use) | [[Citation]](https://github.com/can-can-ya/QPMIL-VL?tab=readme-ov-file#-citation)

**Abstract**: Whole Slide Image (WSI) classification has very significant applications in clinical pathology, *e.g.*, tumor identification and cancer diagnosis. Currently, most research attention is focused on Multiple Instance Learning (MIL) using static datasets. One of the most obvious weaknesses of these methods is that they cannot efficiently preserve and utilize previously learned knowledge. With any new data arriving, classification models are required to be re-trained on both previous and current new data. To overcome this shortcoming and break through traditional vision modality, this paper proposes the first *Vision-Language*-based framework with *Queryable Prototype Multiple Instance Learning* (QPMIL-VL) specially designed for incremental WSI classification. This framework mainly consists of two information processing branches. One is for generating the bag-level feature by prototype-guided aggregating on the instance features. While the other is for enhancing the class feature through class ensemble, tunable vector and class similarity loss. The experiments on four TCGA datasets demonstrate that our QPMIL-VL framework is effective for incremental WSI classification and often significantly outperforms other compared methods, achieving state-of-the-art (SOTA) performance.

<div align="center">
  <a href="https://arxiv.org/pdf/2410.10573">
    <img width="80%" height="auto" src="./docs/QPMIL-VL.png" />
  </a>
</div>

---

### 📚 Recent Updates

- **12/10/2024**: Code & [paper](https://arxiv.org/abs/2410.10573) are live.

*On updating. Stay tuned.*

## 👩‍💻 Running the Code

### 1. CONCH Pre-trained Weight

First of all, you could download the pre-trained weight of pathology VLM CONCH [here](https://huggingface.co/MahmoodLab/CONCH/tree/main).

### 2. WSI Preprocessing

We use [CLAM](https://github.com/mahmoodlab/CLAM) to crop non-overlapping `256 × 256` patches from the segmented tissue at `10×` magnification. Then, pre-trained image encoder in [CONCH](https://github.com/mahmoodlab/CONCH) is used to extract instance features. You could move to [Pipeline-Processing-TCGA-Slides-for-MIL](https://github.com/liupei101/Pipeline-Processing-TCGA-Slides-for-MIL) for a detailed tutorial.

**Of course, you could also use the [dataset](https://pan.baidu.com/s/15ZxpXI2zqVuBb7gsqv_DTA?pwd=gd30) we prepared directly.**

### 3. Running Environment

All experiments are run on a machine with

- two NVIDIA GeForce RTX 3090 GPUs
- python==3.8 and pytorch==1.11.0+cu113

Detailed package requirements:

- for `pip` or `conda` users, full requirements are provided in [requirements.txt](https://github.com/can-can-ya/QPMIL-VL/blob/main/requirements.txt).
- for `docker` users, you could use our base docker image via `docker pull yuukilp/deepath:py38-torch1.11.0-cuda11.3-cudnn8-devel` and then install additional essential python packages (see [requirements.txt](https://github.com/can-can-ya/QPMIL-VL/blob/main/requirements.txt)) in the container.

### 4. Training Models

All important arguments are explained in `configs/main.yaml`. **You could replace the values of `dataset_root_dir` and `conch_ckpt_path` with <u>the root directory of dataset</u> and <u>the path of CONCH pre-trained weight</u> respectively.**

Finally, in the project directory `scripts/`, execute the following command (*ten-fold cross-validation*):

```bash
./main.sh
```

## 🌈 Acknowledgements

Some parts of codes in this repo are adapted from the following amazing works. We thank the authors and developers for their selfless contributions. 

- [Luping Ji](https://faculty.uestc.edu.cn/jiluping/zh_CN/index.htm), [Pei Liu](https://liupei101.github.io): Provide detailed guidance.
- [CONCH](https://github.com/mahmoodlab/CONCH): Our QPMIL-VL is driven by this great pathology VLM.
- [AttriCLIP](https://github.com/vanity1129/AttriCLIP): A prompt tuning approach for incremental learning of natural image based on CLIP.
- [L2P](https://github.com/JH-LEE-KR/l2p-pytorch): A prompt-query mechanism based on the pre-trained ViT model is designed to mitigate catastrophic forgetting in incremental learning for natural image.
- [TOP](https://github.com/miccaiif/TOP): Proposes a two-level prompt learning MIL framework based on GPT-4 and CLIP for Few-shot WSI Classification (FSWC) problem.
- [TaskRes](https://github.com/geekyutao/TaskRes): An efficient fine-tuning method of VL pre-training model is proposed.

## 🔓 License and Terms of Use

ⓒ UESTC. The models and associated code are released under the CC-BY-NC-ND 4.0 license and may **only be used for non-commercial, academic research purposes with proper attribution**. Any commercial use, sale, or other monetization of the QPMIL-VL model and its derivatives are prohibited and require prior approval. If you are a commercial entity, please contact the **corresponding author** ([Luping Ji](https://faculty.uestc.edu.cn/jiluping/zh_CN/index.htm)).

## 📝 Citation

If you find this work helps your research, please consider citing our [paper](https://arxiv.org/abs/2410.10573):

Gou, J.; Ji, L.; Liu, P.; and Ye, M. 2024. Queryable Prototype Multiple Instance Learning with Vision-Language Models for Incremental Whole Slide Image Classification. arXiv preprint arXiv:2410.10573.

```txt
@article{gou2024queryable,
  title={Queryable Prototype Multiple Instance Learning with Vision-Language Models for Incremental Whole Slide Image Classification},
  author={Gou, Jiaxiang and Ji, Luping and Liu, Pei and Ye, Mao},
  journal={arXiv preprint arXiv:2410.10573},
  year={2024}
}
```

Additionally, our another work, for the first time, proposes a new *Vision-Language-based Survival Analysis* (VLSA) paradigm. If you find VLSA useful, please refer to the [code](https://github.com/liupei101/VLSA) and also consider citing the corresponding [paper](https://arxiv.org/abs/2409.09369):

Liu, P.; Ji, L.; Gou, J.; Fu, B.; and Ye, M. 2024. Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology. arXiv preprint arXiv:2409.09369.

```txt
@article{liu2024interpretable,
  title={Interpretable Vision-Language Survival Analysis with Ordinal Inductive Bias for Computational Pathology},
  author={Liu, Pei and Ji, Luping and Gou, Jiaxiang and Fu, Bo and Ye, Mao},
  journal={arXiv preprint arXiv:2409.09369},
  year={2024}
}
```