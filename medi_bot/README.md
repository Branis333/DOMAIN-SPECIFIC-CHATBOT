---
language: en
license: mit
tags:
- medical
- healthcare
- chatbot
- question-answering
- symptoms
- diseases
- gpt2
datasets:
- custom-medical-qa
- medquad
- symptom-disease-dataset
metrics:
- perplexity
- loss
model-index:
- name: symptom-gpt2-chatbot
  results:
  - task:
      type: text-generation
      name: Medical Q&A Generation
    metrics:
    - name: Perplexity
      type: perplexity
      value: 1.50
    - name: Final Validation Loss
      type: loss
      value: 0.4024
---

# 🏥 Medical Symptom Chatbot - GPT2 Fine-tuned

A specialized GPT-2 model fine-tuned on medical Q&A data to assist with symptom analysis, disease information, and health-related questions.

## 🎯 Model Description

This model is based on GPT-2 and has been fine-tuned on a comprehensive medical dataset combining:
- **Symptom-Disease mappings** with descriptions and precautions
- **MedQuAD dataset** with expert medical Q&A pairs
- Custom medical knowledge base

**⚠️ IMPORTANT DISCLAIMER:** This model is for informational and educational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.

## 📊 Training Details

### Dataset Statistics
- **Total Training Samples:** 8,437
- **Validation Samples:** 938
- **Total Dataset Size:** 9,375 medical Q&A pairs

### Training Configuration
- **Base Model:** GPT-2 (124M parameters)
- **Training Epochs:** 10
- **Batch Size:** 4
- **Learning Rate:** 3e-5
- **Optimizer:** AdamW
- **Max Sequence Length:** 512 tokens
- **Hardware:** NVIDIA GPU (CUDA enabled)
- **Training Time:** ~3.5 hours

### Performance Metrics

| Epoch | Train Loss | Val Loss | Perplexity |
|-------|------------|----------|------------|
| 1     | 0.5518     | 0.4664   | 1.59       |
| 2     | 0.4553     | 0.4366   | 1.55       |
| 3     | 0.4162     | 0.4196   | 1.52       |
| 4     | 0.3865     | 0.4088   | 1.51       |
| 5     | 0.3621     | 0.4015   | 1.49       |
| 6     | 0.3415     | 0.3975   | 1.49       |
| 7     | 0.3233     | 0.3988   | 1.49       |
| 8     | 0.3069     | 0.3984   | 1.49       |
| 9     | 0.2917     | 0.3977   | 1.49       |
| **10** | **0.2781** | **0.4024** | **1.50** |

**Final Model Performance:**
- ✅ Training Loss: **0.2781**
- ✅ Validation Loss: **0.4024**
- ✅ Validation Perplexity: **1.50**

## 🚀 Usage

### Basic Usage

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load model and tokenizer
model_name = "Branis333/symptom-gpt2-chatbot"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Generate response
question = "I have fever and cough. What could this be?"
prompt = f"User: {question} Bot:"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = response.split("Bot:")[-1].strip()
print(answer)
```

## 💡 Example Queries

### Symptom Analysis
```
User: I have fever and cough. What could this be?
Bot: You may be experiencing a respiratory infection...
```

### Disease Information
```
User: What are the symptoms of diabetes?
Bot: Common symptoms include increased thirst, frequent urination...
```

## 📁 Dataset Sources

1. **Kaggle Symptom-Disease Dataset** - Disease descriptions, symptom mappings, precautions
2. **MedQuAD** - Expert-curated medical Q&A from multiple domains

## ⚠️ Limitations

1. **Not a Medical Professional**: Cannot replace professional medical advice
2. **Training Data Bias**: Limited to information in training data
3. **Hallucination Risk**: May generate plausible but incorrect information
4. **Language**: Primarily English medical texts

## 🔒 Ethical Considerations

- **Informational Only**: Should not be used for self-diagnosis
- **Professional Consultation Required**: Always seek medical professionals for health concerns
- **Verification**: Cross-check any medical information with reliable sources

## 📄 License

MIT License - Free to use with attribution

---

**Built with ❤️ using Hugging Face Transformers**

*Last Updated: October 2024*
