<p align="center">
  <img src="https://raw.githubusercontent.com/SNOWTEAM2023/GEM/main/materials/logo.jpg" width="400">
</p>

<a href='https://arxiv.org/abs/2511.13007'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

💻 This is the official implementation of paper [**GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs**](https://arxiv.org/abs/2511.13007).

✅ This paper has been accepted by [**The 40th AAAI Conference on Artificial Intelligence (AAAI) 2026**](https://aaai.org/conference/aaai/aaai-26/).

**GEM** is designed for **LLMs alignment at low-resource and domain-specific scenarios**. Instead of training a discriminative reward model on preference data, GEM directly train the LLM to internalize a **closed-loop optimization architecture** that can extract and exploit the multi-dimensional, fine-grained cognitive signals implicit in human preferences.

#### Authors
Yiyang Zhao, Huiyu Bai, [Xuejiao Zhao*](https://zxjwudi.github.io/xuejiaozhao/)

**Nanyang Technological University  &nbsp; | &nbsp; LILY Research Centre (NTU) &nbsp; |&nbsp; ANGEL Research Institute (NTU)**

\* Corresponding author

[![Stargazers repo roster for @SNOWTEAM2023/GEM](https://reporoster.com/stars/SNOWTEAM2023/GEM)](https://github.com/SNOWTEAM2023/GEM/stargazers)

---

## :fire: News
* **[2025.12.3]** We fixed bugs and polished the readme! 🔧😎
* **[2025.12.1]** We release github repository of **GEM**. 💪 Have a try！
* **[2025.11.17]** We release the preprint of **GEM** on [arXiv](https://arxiv.org/abs/2511.13007).
* **[2025.11.08]** Accepted as an **Oral presentation** to AAAI 2026. 🎉

## 🧭 Framework Overview

<p align="center">
  <img src="materials/gem.png" width="1000">
</p>
    <p align="center"><em>Figure 1: Overview of GEM.</em></p >

**GEM** aligns base LLM using human preference data by a **Coginitive Feedback Loop**, which includes **Cognitive Filtering** and **SEGA** modules.

Key modules of GEM include:

- **Cognitive Filtering**: Generate `k` Chain-of-Thought (CoTs) candidates per query and **rank** them by Entropy-guided Token Scoring module. Entropy-guided Token Scoring module encourage **exploration mid‑CoT** (high entropy on top‑m steps) and **confidence at the end** (low final entropy).
- **SEGA**: A **listwise** objective that updates the policy using **group-mean–centered advantages**, which update with weights proportional to **Aᵢ = rᵢ − r̄** within each k-way group.

## 🚀 Quickstart


### 0) Install
```bash
git clone https://github.com/SNOWTEAM2023/GEM.git

cd GEM
pip install -r requirements.txt
```

### 1) Data Preparation

This project expects *preference pairs* of the form:
```jsonl
{"prompt": "...", "chosen": "...", "rejected": "..."}
```
A tiny synthetic set is provided under `data/` to sanity-check the pipeline. For real runs, point to public datasets after you have download permissions. The project primarily utilizes the following two types of datasets for training and evaluation as described in the paper:

1. **General Domain Dataset**: We selected the publicly available ["Skywork-Reward-Preference-80K-v0.2"]("Skywork-Reward-Preference-80K-v0.2") as the base preference data. For few-shot scenarios, we used a small number of high-quality samples (approximately 3,000) for experimentation and tested on public benchmarks such as:

- [UltraFeedback](https://github.com/OpenBMB/UltraFeedback) A large-scale, fine-grained, and diverse preference dataset, containing prompts from various resources, and annotated by GPT-4 in four aspects: instruction following, authenticity, honesty, and usefulness.
  
- [PKU-SafeRLHF](https://github.com/PKU-Alignment/safe-rlhf) A human-annotated preference dataset, containing over 300,000 human-labeled comparison data points, covering preferences for usefulness and harmlessness, aimed at promoting research on the safe alignment of large language models.

- [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench) A dataset for evaluating the capabilities of reward models, covering multiple categories including chat, reasoning, and safety, is designed to test the performance of reward models in complex and structured queries.


2. **Medical Domain Dataset**: To verify the effectiveness of the method in specialized scenarios, the paper constructed a medical preference dataset simulating a low-resource environment based on the [iCliniq](https://www.icliniq.com/) dataset. The dataset consists of 3,500 entries, with 3,000 used for training and 500 for validation. The data is derived from anonymized segments of real clinical conversations and publicly available medical data. It has undergone deduplication, normalization, anonymization, and expert annotation to form a structured preference format of (question, answer_pos, answer_neg).

When reproducing or conducting research using the above datasets, please note the following points:

- The preprocessing and filtering methods for the general domain dataset are detailed in the paper and script comments. It is recommended to ensure that there is no overlap between the training and test sets before training.
- If you have other custom preference data (such as for question-answering or dialogue scenarios), you can also integrate it into the same process in the format of (prompt, chosen, rejected).

### 2) Run GEM：

```bash
python GEM.py
```

### ✨ Code Structure
The code structure and corresponding comments of this repository are as the following:

```
GEM/
├── GEM.py                      # Main entry script for running GEM
├── data/                       # Example preference data
│   └── preference_data.jsonl
│
├── src/                        # Core implementation of GEM
│   ├── __init__.py
│   ├── config.py               # Configuration utilities
│   ├── dataset.py              # Dataset & dataloader definitions
│   ├── entropy_scorer.py       # Entropy-based scoring
│   ├── gem_trainer.py          # GEM training pipeline
│   └── sft_trainer.py          # Supervised fine-tuning (SFT) trainer
│
├── materials/                  # Figures & assets for the paper
├── README.md                   # Project introduction and usage
├── LICENCE.txt                 # Licence information
└── requirements.txt            # Python dependencies
```


### ✨ Implementation Notes

- **Entropy-guided scoring** implements: *final-answer entropy penalty* and *top‑m fork entropies* average per Eq. (1).
- **SEGA** implements group-mean baseline and advantage weighting per Eq. (2) with `wᵢ ∝ Aᵢ`. We provide `identity` or `softmax` mapping from score → reward.
- **Cognitive filtering** also supports optional trimming of low-scoring outliers and pairing top/bottom candidates before optimization.
- The provided **evaluation** uses `r(q,a)=β·log πθ(a|q)` for two-way comparisons.

> *Note: See comments in code for per‑step references back to the paper’s sections, equations, and figures.

### ✨ Reproducibility Knobs
- `k`: number of CoT candidates per query
- `lambda_fork (λ)`: weight for fork entropy term in Eq. (1)
- `top_m`: fraction or count for top‑entropy tokens used in Eq. (1)
- `reward_mapping`: `identity` or `softmax`
- `beta`: implicit reward scale in `r(q,a)`

---

## 📊 Experimental Results

<p align="center">
  <img src="materials/result1.png" width="630">
</p>

*Table 1:Preference-prediction accuracy (%). Higher is better, and the best performing method in each experiment is in bold and the second-best method is indicated with underlining*


<p align="center">
  <img src="materials/result2.png" width="380">
</p>

*Table 2: Agreement with medical-expert preferences on the 500-sample validation set.*

<p align="center">
  <img src="materials/result3.png" width="750">
</p>


*Table 3: Down-stream task results. Accuracy (%) for GSM8K / MATH, exact-match (%) for TruthfulQA; MT-Bench reports win-rate (%) against the SFT baseline.*

---

## 📖 Citation

If you find GEM helpful in your research, please cite our paper:

```bibtex
@article{zhao2025gem,
  title={GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs},
  author={Zhao, Yiyang and Bai, Huiyu and Zhao, Xuejiao},
  journal={arXiv preprint arXiv:2511.13007},
  year={2025}
}
```

## 🔑 License
This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).
Commercial use is prohibited without a separate license agreement with the author.



