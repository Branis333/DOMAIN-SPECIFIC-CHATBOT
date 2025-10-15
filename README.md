# Medical Symptom Chatbot

A domain-specific chatbot for medical symptom analysis and health information using GPT-2 fine-tuning and Gradio interface.

## Project Overview

This project implements a medical chatbot that can:
- Answer questions about symptoms and diseases
- Provide general medical information
- Suggest possible conditions based on symptoms
- Offer basic precautionary measures
- Guide users while maintaining appropriate medical disclaimers

## Project Structure

```
DOMAIN-SPECIFIC-CHATBOT/
├── medi_bot/
│   ├── app.py                  # Gradio web interface
│   ├── updated.ipynb          # Training notebook
│   ├── train.csv              # Training dataset
│   ├── val.csv                # Validation dataset
│   ├── full_dataset.csv       # Complete dataset
│   ├── MedQuAD/              # Medical Q&A Dataset
│   └── models/                # Trained model checkpoints
```

## Technologies Used

- **Python Libraries:**
  - PyTorch - Deep learning framework
  - Transformers - Hugging Face's transformers library
  - Gradio - Web interface creation
  - Pandas - Data manipulation
  - XML ElementTree - XML parsing

- **Model:**
  - Base: GPT-2 (124M parameters)
  - Fine-tuned on medical Q&A data

## Dataset

The chatbot is trained on a combination of:
1. Symptom-Disease Dataset
   - Disease descriptions
   - Symptom mappings
   - Recommended precautions
   - Severity indicators

2. MedQuAD (Medical Question Answering Dataset)
   - Expert-curated medical Q&A pairs
   - Multiple medical domains coverage
   - High-quality medical information

Total dataset size: 9,375 Q&A pairs
- Training set: 8,437 samples
- Validation set: 938 samples

## Training Details

- **Model Configuration:**
  - Training Epochs: 10
  - Batch Size: 4
  - Learning Rate: 3e-5
  - Optimizer: AdamW
  - Max Sequence Length: 512 tokens

- **Performance Metrics:**
  - Final Training Loss: 0.2781
  - Final Validation Loss: 0.4024
  - Final Perplexity: 1.50

## Usage

1. Install required packages:
```bash
pip install torch transformers gradio pandas
```

2. Run the chatbot:
```bash
python medi_bot/app.py
```

3. Access the web interface at `http://localhost:7860`

## Sample Interactions

```
User: "I have fever and cough. What could this be?"
Bot: Provides possible conditions and relevant medical information

User: "What are the symptoms of diabetes?"
Bot: Lists common diabetes symptoms and basic information

User: "What is hypertension?"
Bot: Explains hypertension and its basic characteristics
```

## Important Disclaimers

⚠️ This chatbot is for educational and informational purposes only:
- Not a replacement for professional medical advice
- Always consult healthcare professionals for medical concerns
- Information should be verified with reliable medical sources
- Not suitable for emergency medical situations

## Future Improvements

- Expand the training dataset with more medical sources
- Implement more sophisticated medical entity recognition
- Add multilingual support
- Enhance response accuracy for specific medical domains
- Integrate medical knowledge graphs for better context

## Author

- Name: Branis333
- Project: Domain-Specific Chatbot (Medical)
- Model Available: [Hugging Face Hub](https://huggingface.co/Branis333/symptom-gpt2-chatbot)
- Model space: [Hugging Face spaces](https://huggingface.co/spaces/Branis333/symptom_gpt2)
## License

This project is licensed under the MIT License - see the LICENSE file for details.
