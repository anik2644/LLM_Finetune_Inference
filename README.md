# DU TSC Domain-Specific Question Answering with T5

This repository contains a fine-tuned **T5-Base** model capable of answering domain-specific questions related to the *Teacher-Student Center (TSC), University of Dhaka (DU)*.  
The model is trained entirely on a custom DU-TSC dataset prepared for question–answering tasks.

---

## 🔍 Overview

The purpose of this project is to adapt the pre-trained **T5-Base** model from Hugging Face to understand and answer queries specifically about DU TSC. After fine-tuning, the model is capable of responding with high accuracy to questions regarding:

- Academic programs  
- Departments and units  
- TSC facilities and services  
- Students’ information  
- DU-TSC administrative and contextual information  

This makes the model suitable for chatbots, automated FAQ systems, and domain-specific assistants.

---

## 📘 Dataset

The model is trained on a high-quality DU-TSC domain-specific Q&A dataset.

📌 **Dataset Link (Hugging Face):**  
https://huggingface.co/datasets/mhdank/tsc_du_data

The dataset includes:

- Domain-specific question–answer pairs  
- DU TSC academic and administrative information  
- Student service–related answers  
- Cleaned and structured text suitable for T5 text-to-text training  

---

## 🤖 Model

The fine-tuned model is based on:

- **Pretrained model:** `t5-base` (Hugging Face)
- **Training approach:** Text-to-Text (Sequence-to-Sequence)
- **Task:** Domain-specific Question Answering

### 🔧 Fine-Tuning Steps

1. **Dataset Preparation**  
   The dataset was transformed into text-to-text format:



## 🤗 Hugging Face Model Repository  
👉 **https://huggingface.co/mhdank/testing_checkpoint_new**

---

## 🛠️ Requirements

You will need:

- Python 3.x  
- Hugging Face Transformers  
- Datasets  
- PyTorch  
- (Optional) Accelerate  
- Jupyter Notebook / Colab  

Install dependencies:

```bash
pip install -r requirements.txt
