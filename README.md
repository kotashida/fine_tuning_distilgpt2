# Fine-Tuning a Language Model for Question Answering on Limited Hardware

This repository contains the code and instructions to fine-tune a Large Language Model (LLM) for question answering, specifically designed to run on hardware with limited resources.

**Overview**

This project demonstrates how to fine-tune a smaller LLM for question-answering tasks using a custom dataset. We address the challenge of training a model with limited resources by selecting an appropriate model, optimizing training parameters, and leveraging efficient libraries. This README guides you through the setup, data preparation, fine-tuning, and usage of the model.

**Dataset**

The model was fine-tuned on a dataset derived from the **Apple 2025 Proxy Statement**. This dataset consists of questions and corresponding answers extracted from the document, formatted as a JSONL file named `fine_tuning_dataset.jsonl`. You can replace this with your own JSONL file (details below).

**Key Features**

*   **Hardware-Conscious:** Designed to work within the constraints of an integrated GPU and limited RAM.
*   **Efficient Model Selection:** Used DistilGPT-2, which is smaller and faster than larger LLMs.
*   **Step-by-Step Guide:** Provides detailed instructions for setting up the environment, preparing the data, and fine-tuning the model.
*   **Easy Inference:** Simple function to generate answers to new questions.

**Table of Contents**

1.  [Prerequisites](#prerequisites)
2.  [Setup](#setup)
3.  [Data Preparation](#data-preparation)
4.  [Fine-Tuning](#fine-tuning)
5.  [Inference](#inference)
6.  [Custom Dataset](#custom-dataset)
7.  [Limitations and Expectations](#limitations-and-expectations)
8.  [Troubleshooting and Tips](#troubleshooting-and-tips)

## Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   Basic familiarity with command-line interfaces.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    ```bash
    venv\Scripts\activate  # For Windows
    ```
    
4.  **Install required packages:**

    ```bash
    pip install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html transformers datasets
    ```
    *(Note: This installs the CPU version of PyTorch. Adjust the version if needed)*

## Data Preparation

1.  **Ensure your dataset is in the correct format:**
   *   The dataset should be in a JSONL file (e.g., `fine_tuning_dataset.jsonl`).
    *  Each line in the file should be a JSON object containing a `text` field (the question) and an `answer` field (the corresponding answer):
    ```json
    {"text": "[INST] What are some of the services that Apple provides? [/INST]", "answer": "Apple Pay, Apple Music, Apple TV+."}
    ```

2.  **Place the dataset file** (`fine_tuning_dataset.jsonl` or a name of your choosing) in your project directory.

## Fine-Tuning

1.  **Run the fine-tuning script:**
     *   This involves loading a pre-trained model and training it on your data:

    ```bash
    python training.py
    ```
    
    *  The default model is set to distilgpt2. To change the model, edit the variable `model_name` in `training.py` to one of the suggested models (e.g., albert-base-v2, mistralai/Mistral-7B-v0.1).

2.  **The fine-tuned model will be saved** in the `fine-tuned-model` directory after the training is complete.

## Inference

1.  **After training, you can generate answers to new questions** using the `generate_answer` function provided:

     ```bash
    python inference.py
    ```
    *   *(Note: The `inference.py` script loads the trained model and tokenizer as necessary.)*
    *   You can change the input query provided to this function within the `inference.py` file.

## Custom Dataset

To use your own dataset:

1.  **Format** your data into a JSONL file as described in [Data Preparation](#data-preparation).
2.  **Update the `data_files`** argument in the `training.py` file to point to your file instead of `fine_tuning_dataset.jsonl`.

## Limitations and Expectations

*   **Training Speed:** Expect slow training times when using a CPU for training.
*   **Model Performance:** The fine-tuned model may not perform as well as models trained on more powerful hardware.
*   **Resource Constraints:** Integrated GPUs offer minimal acceleration; training relies primarily on the CPU.

## Troubleshooting and Tips

*   **Out-of-Memory Errors:**
    *   Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps` in the training arguments within `training.py`.
*   **Slow Training:**
    *   Start with a small subset of the dataset to verify the setup is correct.
*   **Monitor Resources:**
    *   Use Task Manager to monitor CPU and memory usage.
*   **Regular Saving:**
    *   Use `save_steps` in `TrainingArguments` to save checkpoints during training.

By following these steps, you can successfully fine-tune a language model for question answering using limited resources.
