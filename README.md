---

# TinyLlama Specialized LLM for Sleep & Stress Management

This repository contains the complete pipeline to fine-tune, specialize, prune, and benchmark the **TinyLlama** model for healthcare advice in sleep and stress management. The workflow includes **two-stage training**, **LoRA adapters**, **final pruning**, and **model benchmarking** with both classical metrics and expert LLM judges.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Stage 1: Continued Pre-training](#stage-1-continued-pre-training)
5. [Stage 2: Domain-Specific LoRA Fine-Tuning & Pruning](#stage-2-domain-specific-lora-fine-tuning--pruning)
6. [Benchmarking and Evaluation](#benchmarking-and-evaluation)
7. [Inference Examples](#inference-examples)
8. [Outputs](#outputs)

---

## Project Overview

The main goals of this project are:

* Fine-tune a **TinyLlama 1.1B** model for the healthcare domain (sleep and stress management).
* Inject domain-specific knowledge using **LoRA adapters**.
* Perform **safe zero-pruning** to reduce model size while preserving performance.
* Benchmark the final model using **semantic similarity**, **ROUGE-L**, and **LLM judges**.
* Compare against baseline models like Phi-3-mini, Gemma, Falcon, and OPT.

---

## 🧠 Algorithms Used

| Category                    | Algorithm / Technique                                | Purpose                                                                                                     |
| --------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **1. Language Modeling**    | **TinyLlama (LlamaForCausalLM architecture)**        | Transformer-based autoregressive model used as the base architecture for conversational fine-tuning.        |
| **2. Fine-Tuning Strategy** | **Instruction Fine-Tuning (Supervised Fine-Tuning)** | Aligns the model to follow medical Q&A instructions for sleep and stress queries.                           |
| **3. Model Compression**    | **Pruning Algorithm (Weight Magnitude Pruning)**     | Removes less important model weights to reduce size and improve efficiency without major accuracy loss.     |
| **4. Evaluation**           | **Benchmarking & Domain Evaluation**                 | Compares model responses using domain-specific test cases and quality metrics (fluency, relevance, safety). |
| **5. Tokenization**         | **SentencePiece BPE Tokenization**                   | Efficient text tokenization compatible with Llama models.                                                   |
| **6. Optimization**         | **AdamW Optimizer**                                  | Used during fine-tuning for stable convergence and better generalization.                                   |
| **7. Data Processing**      | **JSON-based Dataset Parsing and Normalization**     | Handles the sleep–stress dataset for Q&A and dialogue pairs.                                                |

## 🧰 Tools and Frameworks Used

| Tool / Library                | Purpose                                                                    |
| ----------------------------- | -------------------------------------------------------------------------- |
| **Hugging Face Transformers** | Model loading, fine-tuning, pruning, and saving in `.safetensors` format.  |
| **PyTorch**                   | Backend deep learning framework for model training and optimization.       |
| **Accelerate / BitsAndBytes** | Efficient GPU memory utilization and low-bit precision quantization.       |
| **Pandas & NumPy**            | Dataset preprocessing and analysis.                                        |
| **Matplotlib**                | Visualization of training metrics and benchmarking results.                |
| **Google Colab / Kaggle**     | Training and pruning environment (GPU-enabled).                            |
| **Hugging Face Hub**          | Model hosting and distribution.                                            |
| **Jupyter Notebook (.ipynb)** | Used for stepwise experimentation, training, and evaluation documentation. |


## 🧩 Methods Implemented

| Method                                | Description                                                                                                             |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Continued Pretraining (Stage 1)**   | Exposes the TinyLlama model to domain-specific (medical/sleep) corpus to adapt vocabulary and contextual understanding. |
| **Instruction Fine-Tuning (Stage 2)** | Uses labeled Q&A datasets to train the model on patient-centered dialogue generation.                                   |
| **Model Pruning**                     | Removes redundant parameters using structured/unstructured pruning for smaller, faster models.                          |
| **Domain Benchmarking**               | Compares model performance on healthcare-related prompts and tracks metrics in `leaderboards_full.json`.                |
| **Privacy-Preserving Deployment**     | Model designed to run locally (offline inference), ensuring user data privacy.                                          |
| **Prompt Template Integration**       | `chat_template.jinja` defines the conversational prompt format for consistent dialogue behavior.                        |



## Environment Setup

```bash
pip install --upgrade \
    transformers==4.57.0 \
    datasets==2.13.1 \
    accelerate==0.20.3 \
    peft==0.17.1 \
    bitsandbytes \
    sentence-transformers \
    wandb \
    rouge-score
```

* Use **CUDA GPU** if available.
* Python >= 3.8 recommended.
* Optional: **WANDB** login for experiment tracking.

---

## Data Preparation

* Input data should be in **JSONL format** with fields:

```json
{
  "instruction": "Patient has trouble sleeping due to stress; suggest a nightly routine.",
  "input": "",
  "output": "Keep consistent bedtime, limit screens 1 hour before bed, do 10 minutes of breathing exercises."
}
```

* A **single JSONL** or separate train/val JSONLs can be used.
* Ensure **unlearning prompts** are added to prevent the model from generating unsafe instructions.

---

## Stage 1: Continued Pre-training

This stage builds foundational medical knowledge:

1. Load **TinyLlama 1.1B** with **custom medical tokenizer**.
2. Pre-train on a corpus of **medical textbooks**.
3. Apply **LoRA adapters** for efficient fine-tuning.
4. Save adapters to `tinyllama-medical-pretrained`.

Key steps:

```python
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, device_map="auto", torch_dtype=torch.float16, load_in_4bit=True
)
model.resize_token_embeddings(len(tokenizer))

# Apply LoRA adapters
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

* Tokenize dataset and group into **blocks of 1024 tokens**.
* Train for **1 epoch** over the full corpus.
* Save adapters for Stage 2.

---

## Stage 2: Domain-Specific LoRA Fine-Tuning & Pruning

1. Load the **pre-trained/pruned TinyLlama**.
2. Fine-tune with **domain-specific dataset** for sleep and stress.
3. Add **unlearning examples** for unsafe queries.
4. Use **tokenize_and_mask** to train only on response tokens.
5. Perform **safe zero-pruning** on the last few attention layers.

Key configuration:

```python
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-5
FINAL_PRUNE_LAST_N_LAYERS = 4
FINAL_PRUNE_HEADS_FRACTION = 0.5
```

* LoRA adapters are merged into a **standalone final model** saved in `final_merged_pruned`.
* Final pruning zeroes out selected attention heads in the last layers.

---

## Benchmarking and Evaluation

**Two approaches are implemented:**

1. **Classical benchmarking**:

   * Uses **semantic similarity** (SentenceTransformer embeddings)
   * **ROUGE-L** scores
   * Measures **latency** and **memory usage**

2. **Judge LLM benchmarking**:

   * Uses expert LLMs (MedGemma 4B, Meditron 7B)
   * Labels model outputs as `consistent` / `hallucinated` / `unknown`
   * Produces **leaderboards** across all models

```python
results = benchmark_model(model_name, dataset, batch_size=8)
```

* Leaderboards are stored in JSONL files (`leaderboards_partial.json`, `leaderboards_full.json`).
* Clean Hugging Face cache and free GPU memory between runs.

---

## Inference Examples

```python
questions = [
    "Patient has mild insomnia and anxiety. Suggest a nightly routine.",
    "Patient complains of stress affecting work. Suggest daily stress management steps."
]

for q in questions:
    print(generate_answer(final_model, tokenizer, q))
```

* Prompt format:

```
Instruction:
Read the following patient scenario and provide a clear, practical answer.
- Use numbered steps or bullets ONLY if multiple steps required.
- Otherwise, provide a concise sentence.
- Do NOT repeat the question.

Patient Scenario:
<instruction text>

Answer:
```

---

## Outputs

* **Adapters**: `tinyllama-medical-pretrained/`
* **Final pruned model**: `final_merged_pruned/`
* **Leaderboards**: `leaderboards_partial.json`, `leaderboards_full.json`

---

## Notes

* Ensure **GPU memory >12GB** for 1.1B models.
* Stage 1 is memory-intensive; **4-bit quantization** reduces usage.
* Stage 2 can run on smaller GPUs with gradient accumulation.
* Unlearning prompts prevent unsafe or irrelevant output.
* Final pruning reduces attention head usage without significant performance loss.

## 🔗 Model Link

You can find the fully fine-tuned and pruned model hosted on Hugging Face here:  
👉 **[ZenBot on Hugging Face](https://huggingface.co/DemonC/ZenBot)**

This link contains:
- `config.json`, `tokenizer.json`, and `model.safetensors`
- Pruned and optimized weights for local inference
- Compatible `chat_template.jinja` for chat-based applications

---


## Contributors
- Subramanian G - [GitHub Profile](https://github.com/Demoncyborg07)
- Teammate 1 - [Thilak L](https://github.com/thilak0105)
- Teammate 2 - [Raghul A R](https://github.com/a-steel-heart)
- Teammate 3 - [Badre Narayanan R G](https://github.com/Badrergb)

