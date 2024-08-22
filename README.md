

# BLIP-2 Chart Summarization and Visual Question Answering (VQA)

## Project Overview

This repository contains the code and models for fine-tuning and evaluating the BLIP-2 model on two specific tasks:
1. **Chart Summarization**: Generating textual summaries from chart images using the Chart-to-Text dataset.
2. **Visual Question Answering (VQA)**: Answering questions based on chart images using the ChartQA dataset.

## Datasets

The following datasets were used in this project:

- **ChartQA Dataset**: [Link to Dataset](https://drive.google.com/drive/folders/1i-H5H8kokxXtNxRfwF4Pu0XTR28Ktb8W?usp=sharing)
)
- **Chart-to-Text Dataset**: [Link to Dataset](https://drive.google.com/drive/folders/1Sdy-n_IzxCWpFDK0nQGk8rWlEOimgh6w?usp=sharing)

## Model Files

The pre-trained and fine-tuned models are stored in a OneDrive directory accessible only to QMUL account holders:

- **[Model Files on OneDrive](https://qmulprod-my.sharepoint.com/:f:/g/personal/ec23817_qmul_ac_uk/EoLMWGYgfq5DrCVVCxSwoycBJgs0JTDggn9pV99STLZ-xw?e=ZgvLL5)**

## Project Structure

The repository is organized as follows:

```plaintext
blip/
├── chartqa/                     # Directory containing ChartQA dataset files
├── imgs/                        # Directory containing example images
├── multiColumn/                 # Directory containing multi-column dataset files
├── out/                         # Directory for output files
├── last_model_c2t.pth           # Best fine-tuned model for chart summarization
├── vqa_last_model.pth           # Last fine-tuned model for VQA
├── Blip2_Chart2text.ipynb       # Notebook for chart summarization
├── Blip2_VQA.ipynb              # Notebook for VQA on chart data
├── chart-to-text_train1.json    # JSON file for training data (chart-to-text)
├── chart-to-text_val1.json      # JSON file for validation data (chart-to-text)
├── chart-to-text_test1.json     # JSON file for test data (chart-to-text)

```

## Running the Notebooks

You can run the Jupyter notebooks provided to fine-tune and evaluate the BLIP-2 model:

### 1. Chart Summarization

Notebook: `Blip2_Chart2text.ipynb`

**Steps to run:**
1. Open the notebook in Jupyter.
2. Ensure all paths to the datasets are correctly specified.
3. Execute the cells to fine-tune and evaluate the model on the Chart-to-Text dataset.

### 2. Visual Question Answering (VQA)

Notebook: `Blip2_VQA.ipynb`

**Steps to run:**
1. Open the notebook in Jupyter.
2. Ensure all paths to the datasets are correctly specified.
3. Execute the cells to fine-tune and evaluate the model on the ChartQA dataset.

## GPU Requirements

Given the large size of the BLIP-2 model and the extensive training times, it is recommended to run these notebooks on a high-performance GPU, such as an A100. The model requires significant memory, and training can take several hours per epoch.

## Using the Model on Individual Images

### Visual Question Answering (VQA)

To use the fine-tuned model for VQA on individual chart images, use the following code:

```python
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load the BLIP2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the fine-tuned weights
model.load_state_dict(torch.load('vqa_last_model.pth'))

# Set the model to evaluation mode
model.eval()

# Example question and image
question = ""  # Replace with your query
image_path = ""  # Replace with the path to your image

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
inputs = processor(text=question, images=image, return_tensors="pt").to(device)

# Make the prediction
generated_ids = model.generate(**inputs, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
```

### Chart Summarization

To generate a summary for a single chart image using the fine-tuned model, use the following code:

```python
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load the BLIP2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the fine-tuned weights
model.load_state_dict(torch.load('last_model_c2t.pth'))

# Load and preprocess the image
image_path = ""  # Replace with the path to your image
image = Image.open(image_path).convert("RGB")
image = image.resize((512, 512))
inputs = processor(images=image, return_tensors="pt").to(device)

# Generate the summary
generated_ids = model.generate(**inputs, max_length=300, num_beams=7)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print("Generated Summary: ", generated_text)
```


This README provides clear instructions for running your code, along with the necessary context for users to understand and use your models effectively. Make sure to update the links to the datasets and OneDrive models before sharing the repository.
