# Qwen2.5-0.5B Alignment via Direct Preference Optimization (DPO)
## University Large Language Models — Lab 10 Demonstration Dashboard

A production-grade, interactive Streamlit dashboard demonstrating and evaluating the alignment of **Qwen/Qwen2.5-0.5B** using **Direct Preference Optimization (DPO)** and **LoRA** (Low-Rank Adaptation) on the **UltraFeedback Binarized** preference dataset.

---

## 🌟 Executive Summary & Experiment Overview

- **Base Model:** `Qwen/Qwen2.5-0.5B`
- **Preference Dataset:** `trl-lib/ultrafeedback_binarized`
- **Alignment Method:** Direct Preference Optimization (DPO)
- **Parameter-Efficient Tuning:** LoRA (PEFT, rank $r=16$, $\alpha=32$, dropout $0.05$)
- **Compared Configurations:**
  1. **Base:** `Qwen/Qwen2.5-0.5B` (unaligned base checkpoint)
  2. **DPO $\beta=0.1$:** LoRA adapter trained with weak reference constraint ($\beta=0.1$)
  3. **DPO $\beta=0.5$:** LoRA adapter trained with strong reference constraint ($\beta=0.5$)
- **Hugging Face Model Repository:** [`ullahhafeez321/qwen-dpo-lab10`](https://huggingface.co/ullahhafeez321/qwen-dpo-lab10)
- **GitHub Repository:** [`ullahhafeez321/Qwen_Model_Alignment`](https://github.com/ullahhafeez321/Qwen_Model_Alignment)

---

## 🔬 Core Concepts: Direct Preference Optimization (DPO)

Direct Preference Optimization (Rafailov et al., 2023) eliminates the need for training a separate reward model or using complex reinforcement learning loops (such as PPO). Instead, DPO derives an exact closed-form mapping between the optimal policy and the underlying reward function:

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

The DPO loss function directly trains the policy network $\pi_\theta$ on preference pairs $(x, y_w, y_l)$ where $y_w$ is the chosen (preferred) completion and $y_l$ is the rejected completion:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### The Role of the $\beta$ Parameter
- **$\beta$ (Beta)** regulates the strength of the Kullback-Leibler (KL) divergence constraint with respect to the reference model $\pi_{\text{ref}}$.
- **$\beta = 0.1$ (lower constraint):** Allows the model greater freedom to steer away from the base policy to satisfy preference distinctions, but increases vulnerability to distribution shift or degradation of general language abilities.
- **$\beta = 0.5$ (higher constraint):** Enforces a tighter penalty on deviating from the base model, preserving base capabilities while applying a more conservative alignment update.
- **Critical Note:** Beta is not a simple "quality dial"; higher beta is not strictly better or worse. It determines the policy-reference divergence trade-off.

---

## 📁 Repository Structure

```
.
├── app.py                      # Main entrypoint and navigation controller
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Complete documentation and lab report
├── .streamlit/
│   └── config.toml             # Custom theme settings (Navy/Blue academic palette)
├── data/
│   ├── base_outputs.csv                # 10 evaluation responses from base model
│   ├── beta_0_1_outputs.csv            # 10 responses from DPO beta=0.1
│   ├── beta_0_5_outputs.csv            # 10 responses from DPO beta=0.5
│   ├── side_by_side_comparison.csv     # Complete 3-way response alignment
│   ├── human_ratings_template.csv      # Human evaluation rubric template
│   ├── human_ratings.csv               # Live/saved human ratings store
│   ├── training_metrics_beta_0_1.json  # Final training metrics (beta=0.1)
│   ├── training_metrics_beta_0_5.json  # Final training metrics (beta=0.5)
│   ├── training_metrics_summary.csv    # Comparative training metrics table
│   ├── trainer_state_beta_0_1.json     # 128-step loss & reward logs (beta=0.1)
│   ├── trainer_state_beta_0_5.json     # 128-step loss & reward logs (beta=0.5)
│   ├── adapter_config_beta_0_1.json    # LoRA PEFT configuration (beta=0.1)
│   └── adapter_config_beta_0_5.json    # LoRA PEFT configuration (beta=0.5)
├── pages/
│   ├── __init__.py
│   ├── overview.py             # Section 1: Academic background & beta explanation
│   ├── experiment_setup.py     # Section 2: Verified LoRA, dataset, and training configs
│   ├── prompt_comparison.py    # Section 3: Side-by-side prompt responses with filters
│   ├── interactive_testing.py  # Section 4: Live GPU/CPU inference with error handling
│   ├── training_analysis.py    # Section 5: Step-level loss, accuracy, and reward curves
│   ├── human_evaluation.py     # Section 6: Interactive 1–5 scoring on 4 criteria
│   ├── conclusion.py           # Section 7: Evidence-based answers to lab questions
│   └── checklist.py            # Section 8: Visual checklist mapping to lab rubric
└── utils/
    ├── __init__.py
    ├── data_loader.py          # Cached data loaders and CSV persistence
    ├── model_loader.py         # Lazy PyTorch/PEFT loading and token decoding
    └── theme.py                # Academic navy/blue theme styling and metric cards
```

---

## 📊 Recorded Training Metrics & Observations

Both fine-tuning runs completed 128 optimization steps (1.0 full epoch) on single-batch causal LM training:

| DPO Beta ($\beta$) | Final Loss | Training Runtime (s) | Samples/sec | Steps/sec | Final Reward Margin | Final Preference Acc |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.1** | 0.681721 | 383.42 | 2.671 | 0.334 | +0.0539 | 66.25% |
| **0.5** | 0.659785 | 385.74 | 2.655 | 0.332 | +0.2090 | 71.25% |

### Key Analysis of Empirical Results
1. **Loss Progression:**
   Both models started around theoretical binary random guessing loss ($\mathcal{L} \approx \ln 2 \approx 0.6931$). $\beta=0.5$ achieved a lower final training loss ($0.6598$) and higher margin ($+0.209$) because the higher $\beta$ scale sharpens the implicit reward differentiation.
2. **Loss vs. Generation Quality:**
   Lower training loss does **not** directly imply superior textual output. DPO training with small parameters ($0.5\text{B}$) for a single epoch exhibits subtle generational alterations. In several instances (e.g., Prompt 7 HTTP/HTTPS explanation and Prompt 5 math word problem), the models carry forward base hallucination artifacts or introduce minor deviations.
3. **Task Type Responsiveness:**
   - **Advice & Study Planning:** Displayed notable structuring refinements with bullet lists and headers.
   - **Reasoning & Arithmetic:** Still exhibited systematic arithmetic calculation failures across both base and aligned adapters, confirming that short DPO does not fix underlying reasoning deficits without chain-of-thought or specialized verifiers.

---

## 🚀 Installation & Local Running

### Prerequisites
- Python 3.10+
- (Optional for live inference): NVIDIA GPU with CUDA or Apple Silicon Mac (MPS), or minimum 4GB system RAM for CPU inference.

### 1. Clone the repository
```bash
git clone https://github.com/ullahhafeez321/Qwen_Model_Alignment.git
cd Qwen_Model_Alignment
```

### 2. Set up virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
For **Dashboard-Only Mode** (fast, zero model download, complete evidence exploration):
```bash
pip install -r requirements.txt
```

For **Live Interactive Inference** (loads base model + LoRA adapters):
```bash
pip install torch transformers peft huggingface_hub accelerate
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment Instructions

### Option A: Streamlit Community Cloud (Recommended)
1. Fork or push this repository to your GitHub account (`ullahhafeez321/Qwen_Model_Alignment`).
2. Log into [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"**, select your repository, branch (`main`), and set the main file path to `app.py`.
4. In Advanced Settings, add optional environment variables or secrets:
   ```toml
   HF_REPO_ID = "ullahhafeez321/qwen-dpo-lab10"
   # HF_TOKEN = "your_optional_read_token"
   ```
5. Click **Deploy**. The app runs in dashboard mode immediately with zero setup.

### Option B: Vercel + Separated Python Inference Architecture
Vercel is designed for serverless frontend execution and cannot host PyTorch/Transformers models (due to 250MB lambda limits and timeout constraints). If a Vercel deployment is desired:
- Deploy the frontend via Next.js on Vercel.
- Deploy the Python inference backend (FastAPI / modal / RunPod / Hugging Face Spaces) on a GPU/CPU enabled host.
- Store `HF_TOKEN` and API endpoints in server environment variables.

---

## ✅ University Lab 10 Rubric Mapping

| Lab 10 Requirement | App Location | Verification Status |
|:---|:---|:---:|
| **Original Prompts ($\ge 5$, $\ge 3$ task types)** | Page 3: Prompt Comparison | ✅ Complete (10 prompts across 6 task types) |
| **Base Model Outputs** | Page 3: Prompt Comparison | ✅ Complete (`base_outputs.csv`) |
| **Chosen/Rejected Examples Inspection** | Page 7: Conclusion & Page 8 | ⚠️ Dataset documented; raw pair visualizer marked |
| **DPO Training for $\beta=0.1$** | Page 2 & Page 5 | ✅ Complete (LoRA config, loss, 128-step metrics) |
| **DPO Training for Additional $\beta=0.5$** | Page 2 & Page 5 | ✅ Complete (LoRA config, loss, 128-step metrics) |
| **Training Logs & Metrics** | Page 5: Training Analysis | ✅ Complete (Interactive Plotly curves, accuracy, margins) |
| **Same-Prompt Base vs. Aligned Comparison** | Page 3: Prompt Comparison | ✅ Complete (3-column responsive comparison cards) |
| **Human Rating Rubric (1–5 scale)** | Page 6: Human Evaluation | ✅ Complete (Helpfulness, Relevance, Clarity, Instruction) |
| **$\beta$ Sensitivity Comparison** | Page 5 & Page 7 | ✅ Complete (Empirical training dynamics & output analysis) |
| **Final Evidence-Based Conclusion** | Page 7: Conclusion | ✅ Complete (All 6 core lab questions answered) |

---

## 📜 Academic Integrity & Citation
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.
- Qwen Team, *Qwen2.5: A Foundation Language Model Series*, 2024.
- TRL (Transformer Reinforcement Learning), Hugging Face.
