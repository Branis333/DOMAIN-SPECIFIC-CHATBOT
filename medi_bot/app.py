import gradio as gr
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load your fine-tuned model from Hugging Face
MODEL_NAME = "Branis333/symptom-gpt2-chatbot"

print("Loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"✓ Model loaded successfully on {device}")

def respond(
    message,
    history,
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    """
    Generate response from the medical chatbot
    """
    # Format the prompt with conversation history
    conversation = system_message + "\n\n"
    
    # Add chat history
    for entry in history:
        if isinstance(entry, dict):
            # New format: messages with 'role' and 'content'
            if entry.get('role') == 'user':
                conversation += f"User: {entry['content']}\n"
            elif entry.get('role') == 'assistant':
                conversation += f"Bot: {entry['content']}\n"
        else:
            # Old format: tuples
            user_msg, bot_msg = entry
            conversation += f"User: {user_msg}\nBot: {bot_msg}\n"
    
    # Add current message
    conversation += f"User: {message}\nBot:"
    
    # Tokenize and generate
    inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the bot's response
    if "Bot:" in full_response:
        response = full_response.split("Bot:")[-1].strip()
        # Remove any trailing "User:" if present
        if "User:" in response:
            response = response.split("User:")[0].strip()
    else:
        response = full_response
    
    return response


# Create Gradio ChatInterface
chatbot = gr.ChatInterface(
    respond,
    type="messages",  # Use new message format
    chatbot=gr.Chatbot(
        height=500,
    ),
    textbox=gr.Textbox(
        placeholder="Ask me about your symptoms or medical questions...",
        container=True,  # keep the input container visible so the send button renders
        scale=7
    ),
    title="🏥 Medical Symptom Chatbot",
    description="Ask questions about symptoms, diseases, and medical conditions. This bot is trained on medical Q&A data. For informational purposes only - always consult healthcare professionals.",
    theme="soft",
    examples=[
        ["I have fever and cough. What could this be?"],
        ["What are the symptoms of diabetes?"],
        ["What is hypertension?"],
        ["I have a headache and nausea. What should I do?"],
        ["What are the precautions for common cold?"],
    ],
    cache_examples=False,
    additional_inputs=[
        gr.Textbox(
            value="You are a helpful medical chatbot that provides information about symptoms and diseases. Always recommend consulting a healthcare professional for serious conditions.",
            label="System Message",
            lines=3
        ),
        gr.Slider(
            minimum=50,
            maximum=300,
            value=150,
            step=10,
            label="Max Tokens",
            info="Maximum length of the response"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.7,
            step=0.1,
            label="Temperature",
            info="Higher = more creative, Lower = more focused"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.05,
            label="Top-p (Nucleus Sampling)",
            info="Controls diversity of responses"
        ),
    ],
    submit_btn="Send",
)

# Launch the app
if __name__ == "__main__":
    chatbot.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Makes it accessible externally
        server_port=7860,
    )