# DocExtract-LLM 📄🤖

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2x_Faster_Finetuning-orange.svg)](https://github.com/unslothai/unsloth)
[![Model](https://img.shields.io/badge/Base_Model-Phi--3--Mini--4k-green.svg)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
[![GGUF](https://img.shields.io/badge/Format-GGUF_Q4__K__M-purple.svg)](https://github.com/ggerganov/llama.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DocExtract-LLM** is a high-performance fine-tuning pipeline designed for extracting structured JSON metadata from unstructured document snippets and HTML web pages using **Phi-3-Mini-4k-Instruct** and **Unsloth**. 

By leveraging **QLoRA (4-bit quantization)**, this project achieves 2x faster fine-tuning with significantly reduced VRAM overhead, enabling seamless deployment to local devices via **GGUF** and **Ollama**.

---

## ✨ Features

- **Structured Data Extraction**: Fine-tuned to transform noisy, unstructured web HTML/text into accurate, valid JSON formats.
- **Fast QLoRA Training**: Powered by [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient and accelerated training on free-tier T4 GPUs (Google Colab compatible).
- **Lightweight Base Model**: Built on `unsloth/Phi-3-mini-4k-instruct-bnb-4bit`.
- **GGUF Export & Quantization**: Converts trained models into `Q4_K_M` GGUF files for local deployment with `llama.cpp`.
- **Ollama Integration**: Generates ready-to-use Ollama `Modelfile` for instant local LLM serving.

---

## 📊 Training Performance

The model was fine-tuned over **3 Epochs** (189 steps) with an effective batch size of 8.

| Step | Training Loss |
| :--- | :------------ |
| 25   | `0.4514`      |
| 50   | `0.1502`      |
| 75   | `0.1341`      |
| 100  | `0.1240`      |
| 125  | `0.1164`      |
| 150  | `0.1129`      |
| 175  | `0.1105`      |

---

## 🛠️ Installation & Setup

### Dependencies
Install the required Python packages:

```bash
pip install unsloth trl peft accelerate bitsandbytes datasets transformers
```

---

## 🚀 Pipeline & Usage

### 1. Model Initialization
Load the 4-bit quantized base model using Unsloth:

```python
from unsloth import FastLanguageModel
import torch

model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
```

### 2. Configure QLoRA Adapters
Apply target projections for LoRA fine-tuning:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

### 3. Dataset Format & Training
Format input documents and expected JSON outputs:

```python
import json
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Input formatting
formatted_data = [
    f"### Input: {item['input']}\n### Output: {json.dumps(item['output'])}<|endoftext|>"
    for item in file_data
]
dataset = Dataset.from_dict({"text": formatted_data})

# Trainer execution
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
trainer.train()
```

---

## 🧪 Inference Example

```python
FastLanguageModel.for_inference(model)

prompt = """<|user|> Extract the product information:
<div class='product'><h2>iPad Air</h2><span class='price'>$1344</span><span class='category'>audio</span><span class='brand'>Dell</span></div><|end|><|assistant|>"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

**Output:**
```json
{"name": "iPad Air", "price": "$1344", "category": "audio", "manufacturer": "Dell"}
```

---

## 💾 Export to GGUF & Ollama

Convert and quantize the model to GGUF `Q4_K_M` format:

```python
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
```

### Run with Ollama
Create an Ollama model using the generated `Modelfile`:

```bash
ollama create docexctract-llm -f gguf_model_gguf/Modelfile
ollama run docexctract-llm "Extract data from: <div class='item'>...</div>"
```

---

## 📂 Repository Structure

```
DocExtract-LLM/
├── Fine_Tuning_project.ipynb   # Complete Colab notebook for training & GGUF export
└── README.md                   # Project documentation
```

---

## 🤝 Acknowledgments

- [Unsloth AI](https://github.com/unslothai/unsloth) for hyper-optimized LLM fine-tuning scripts.
- [Microsoft Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) for the base instruction model.
- [HuggingFace TRL & PEFT](https://github.com/huggingface/trl) for SFT and LoRA support.
