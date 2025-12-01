# DU TSC Domain-Specific Question Answering with T5

This repository contains a fine-tuned T5 model that can answer domain-specific questions related to the University of Dhaka's TSC (Teacher-Student Center). The model was trained using a custom dataset specifically designed to address queries relevant to the DU TSC domain.

## Overview

The goal of this project is to fine-tune a pre-trained T5 base model to answer queries in a domain-specific context, focusing on the DU TSC. The fine-tuned model can respond to questions about academic programs, departments, facilities, and other DU TSC-related information.

## Dataset

The training dataset consists of domain-specific question-answer pairs, including:

- Academic programs and courses offered at DU TSC.
- Information about departments, faculty members, and student services.
- Other domain-specific information relevant to DU TSC.

The dataset was preprocessed and formatted to be compatible with Hugging Face’s T5 model for fine-tuning.

## Model

The model used in this project is the pre-trained **T5 base model** from Hugging Face, which was fine-tuned on the DU TSC dataset. T5 is a transformer-based model that uses text-to-text transfer learning, where both the input and output are treated as text.

### Fine-tuning process:

1. **Dataset Preparation**: The DU TSC domain-specific dataset was formatted into a question-answer pair structure suitable for T5’s text-to-text architecture.
2. **Training**: The model was fine-tuned using the dataset for several epochs, optimizing for performance on the validation set.
3. **Evaluation**: The model's performance was evaluated based on accuracy and its ability to generate relevant answers for the domain-specific queries.

## Requirements

- Python 3.x
- Hugging Face Transformers library
- PyTorch or TensorFlow
- Other dependencies are listed in the `requirements.txt` file.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/du-tsc-qa-model.git
cd du-tsc-qa-model
