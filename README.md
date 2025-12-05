# SemEval 2026 Task 2: Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

## 📄 Abstract

The **Pytorch version of our SemEval-2026 Task 2 submissions** is found within this repository and addresses three major issues in affective computing: **state prediction**, **change forecasting** and **long-term trajectory prediction**.

We utilize efficient "hybrid" architectures; specifically **the Siamese Network ("Bifurcated Leviathan")**, and custom **loss functions (CCC (Concordance Correlation Coefficient))** to prevent regression to the mean, due to **resource limitations associated with consumer-grade hardware (8GB VRAM)**.


---

## 📂 Repository Structure

```bash
├── data/
│   ├── train_subtask1.csv       # Raw longitudinal user data
│   ├── train_subtask2a.csv      # State-transition data
│   └── train_subtask2b.csv      # Long-term history data
├── notebooks/
│   ├── subtask1.ipynb           # Hybrid DistilBERT + LSTM (Validation: Sliding Window)
│   ├── subtask2a.ipynb          # DeBERTa + Projection MLP + CCC Loss
│   └── subtask2b.ipynb          # Bifurcated Siamese "Leviathan" Model
├── predictions/                 # Output CSVs for submission
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

---

## 🎯 Task Definitions & Methodologies

### 1. Subtask 1: Longitudinal Affect Assessment

**The Task:**
Given a chronological sequence of **m** texts $e_1, e_2, \dots, e_m$, the model must produce **Valence & Arousal (V&A)** predictions for each text: $(v_1, a_1), \dots, (v_m, a_m)$.
* *Constraint:* The test split includes **Unseen Users** (zero-shot generalization) and **Seen Users** (temporal tracking).

**Our Solution: The Hybrid Early-Fusion Model**
* **Architecture:** `distilbert-base-uncased` + `BiLSTM`.
* **Innovation:** Instead of relying solely on text, we implement **Early Fusion**. An explicit `User Embedding` (dim=32) is concatenated with the text embedding *before* temporal processing. This allows the LSTM to condition its memory on the specific user identity.
* **Inference:** Uses a custom `SlidingWindowDataset` to prevent "context starvation" (forgetting history) during testing.

### 2. Subtask 2A: Forecasting State Changes

**The Task:**
Given a sequence of texts and their V&A scores up to time $t$, predict the **immediate next-step change** in Valence and Arousal:
$$
\Delta_1 = v_{t+1} - v_t
$$

**Our Solution: The State-Aware Projector**
* **The Problem:** "The Drowning Problem." High-dimensional text vectors (768-dim) overwhelm low-dimensional scalar inputs (current state $v_t, a_t$).
* **The Fix:** A **Projection MLP** boosts the scalar state features into a higher-dimensional space (64-dim) before fusion.
* **Loss Function:** We replaced MSE (Mean Squared Error) with **CCC Loss**.
    * *Observation:* MSE caused the model to predict "zero change" (flatline) to minimize error.
    * *Result:* CCC forces the model to match the *variance* of the trajectory, improving correlation from **0.39** to **0.64**.

### 3. Subtask 2B: Dispositional (Long-Term) Change

**The Task:**
Predict the change between the **mean observed affect** (past) and the **mean future affect** (future):
$$
\Delta_{\text{avg}} = \text{avg}(v_{t+1:n}) - \text{avg}(v_{1:t})
$$

**Our Solution: The "Bifurcated Leviathan"**
* **Architecture:** A Siamese Network with a shared `deberta-v3-large` backbone.
* **Sampling:** Implements a "Head-Tail" protocol, sampling the first 16 essays (Head) and last 16 essays (Tail) to model long-term drift.
* **Residual Learning:** We inject the arithmetic difference of the raw scores ("Naive Math") into the final layer. The network learns to *refine* this statistical trend rather than deriving it from scratch.
* **Bifurcation:** The network splits immediately after the backbone into separate **Valence** and **Arousal** heads to prevent noisy Arousal gradients from disrupting Valence learning.

---

## 📊 Results & Performance

| Task | Metric | Score (Pearson $r$) | Key Insight |
| :--- | :--- | :--- | :--- |
| **Subtask 1** | Valence (Seen) | **0.7026** | User Embeddings are critical for known users. |
| **Subtask 1** | Arousal (Seen) | **0.5186** | Arousal is notoriously harder to model than Valence using text. |
| **Subtask 2A** | Avg Correlation | **0.64** | CCC Loss outperformed MSE by ~27%. |
| **Subtask 2B** | Valence Change | **0.7031** | Residual learning ("Naive Math") prevents scale collapse. |

---

## 🚀 Setup & Usage

### Prerequisites
* Python 3.10+
* NVIDIA GPU (Minimum 8GB VRAM recommended for training)

### Installation
```bash
# Clone the repository
git clone [https://github.com/YourUsername/SemEval-2026-Task2.git](https://github.com/YourUsername/SemEval-2026-Task2.git)
cd SemEval-2026-Task2

# Install dependencies
pip install torch transformers pandas numpy scipy scikit-learn tqdm
```

### Running the Subtasks
Each subtask is self-contained in its respective notebook for reproducibility.

1.  **Subtask 1:** Open `notebooks/subtask1.ipynb`. This script handles the "Seen/Unseen" user split automatically.
2.  **Subtask 2A:** Open `notebooks/subtask2a.ipynb`. Ensure `CCC_LOSS = True` to replicate our best results.
3.  **Subtask 2B:** Open `notebooks/subtask2b.ipynb`. This implements the `HeadTailDataset` class for long-context sampling.

---

## 🤝 Acknowledgements & Credits

### Originality Statement
The architectures presented here (including the "Leviathan" Siamese network and the Hybrid LSTM-Fusion) are original contributions developed for this competition.

### External Resources
We gratefully acknowledge the open-source community. Specifically, initial data processing patterns and file handling structures were informed by the work of:
* **ThickHedgehog (2025):** *Deep-Learning-project-SemEval-2026-Task-2*. Available at: [GitHub](https://github.com/ThickHedgehog/Deep-Learning-project-SemEval-2026-Task-2).

*Note: While preprocessing logic was inspired by the above, the modeling strategies (Early vs. Late Fusion, usage of LSTM for Subtask 1, and CCC optimization) differ significantly in implementation and topology.*

---

## 📜 Citation

If you use this code or our findings in your research, please cite:

```bibtex
@inproceedings{jumakhan2026longitudinal,
  title={Longitudinal Affective Forecasting: Architectures for Generalization, State Change, and Trajectory Prediction},
  author={Jumakhan, Haseebullah and Assad, Soud and Ahmad, Seyed Abdullah},
  booktitle={Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year={2026}
}
```
