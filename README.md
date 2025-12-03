# SemEval 2026 Task 2: Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays

Project decription:

#  Forecasting & Longitudinal Affect Assessment 

This repository contains code and resources for two tasks focused on understanding and forecasting emotional variation from textual data. Each text item (essay or feeling word) is associated with **Valence & Arousal (V&A)** scores.

## Subtask 1 — Longitudinal Affect Assessment

Given a chronological sequence of **m** texts:

e₁, e₂, …, eₘ

your model must produce **Valence & Arousal (V&A)** predictions for each text:

(v₁, a₁), (v₂, a₂), …, (vₘ, aₘ)


### Dataset Characteristics
- Sequences come from **multiple users**.
- Each text item includes a labeled V&A pair.
- The test split includes:
  1. **Unseen users** — users not seen in training.
  2. **Seen users** — users present in training.

This subtask evaluates a model’s ability to track emotional dynamics over time.

## Subtask 2 — Forecasting Future Variation in Affect

Given a sequence of the first **t** texts and their V&A scores:

(e₁, …, eₜ), (v₁, a₁), …, (vₜ, aₜ)


the system must forecast two types of affective variation:

---

### **2A. State Change**

Predict the **immediate next-step change** in Valence (and Arousal):

\[
\Delta_1 = v_{t+1} - v_t
\]

Measures short-term emotional changes.

---

### **2B. Dispositional Change**

Predict the change between the **mean observed affect** and the **mean future affect**:

\[
\Delta_{\text{avg}} = \text{avg}(v_{t+1:n}) - \text{avg}(v_{1:t})
\]

Measures long-term affective shift.

## Goals of This Repository

- Build models to **predict moment-to-moment emotional states**.
- Develop systems to forecast **short-term and long-term affect changes**.
- Benchmark methods on **seen vs unseen** user generalization.
- Explore transformer-based architectures for affect modeling.


