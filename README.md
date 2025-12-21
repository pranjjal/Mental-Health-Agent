# 🧠 Mental Health Chatbot - Multi-Model Deployment

A Streamlit-based deployment of three fine-tuned Large Language Models (Mistral v3 7B, Llama 3.1 8B, and Phi-4 Conversational) for empathetic mental health support conversations.

## 🎯 Project Overview

This project demonstrates:
- **Domain-adapted LLMs** fine-tuned on Mental Health Chatbot Pairs dataset
- **Parameter-efficient fine-tuning** using LoRA/QLoRA techniques
- **Multi-model deployment** with real-time model comparison
- **Emotion-aware conversations** with empathetic, non-judgmental responses

## ⚠️ Important Disclaimer

**This chatbot is for informational and emotional support purposes only.** It does not provide medical diagnosis or treatment. If you're experiencing a mental health crisis, please contact a professional healthcare provider or emergency services immediately.

**Crisis Resources:**
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741

## 🚀 Features

- **Three Fine-Tuned Models**: Compare responses from Mistral v3 7B, Llama 3.1 8B, and Phi-4
- **Model Comparison Mode**: Generate responses from all models simultaneously
- **Customizable Parameters**: Adjust temperature, top-p, and max tokens
- **Efficient Inference**: 4-bit quantization for running models on consumer hardware
- **Clean UI**: Modern, responsive Streamlit interface
- **Chat History**: Persistent conversation tracking within sessions

## 📋 Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended: 16GB+ VRAM for running all three models)
- 50GB+ free disk space for model downloads
- HuggingFace account with access to your deployed models

## 🛠️ Installation

### 1. Clone or Navigate to the Repository

```bash
cd /Users/pranjal/Desktop/Mental-Health-Agent
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installing PyTorch with CUDA support (if you have a GPU):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Configure Your Models

Edit `app.py` and update the `MODEL_CONFIGS` dictionary with your HuggingFace model paths:

```python
MODEL_CONFIGS = {
    "Mistral v3 7B": {
        "model_id": "your-username/mistral-v3-7b-mental-health",  # Replace this
        ...
    },
    "Llama 3.1 8B": {
        "model_id": "your-username/llama-3.1-8b-mental-health",  # Replace this
        ...
    },
    "Phi-4 Conversational": {
        "model_id": "your-username/phi-4-mental-health",  # Replace this
        ...
    }
}
```

### 5. Authenticate with HuggingFace (if models are private)

```bash
pip install huggingface_hub
huggingface-cli login
```

Enter your HuggingFace token when prompted.

## 🎮 Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### Basic Usage

1. **Select a Model**: Choose from the sidebar which model you want to use
2. **Adjust Parameters**: Fine-tune temperature, top-p, and max tokens
3. **Start Chatting**: Type your message in the input box at the bottom
4. **View Response**: The model will generate an empathetic response

### Comparison Mode

1. Enable **"Enable Model Comparison"** in the sidebar
2. Type your message
3. See side-by-side responses from all three models
4. Compare tone, empathy, and response quality

### Clear Chat History

Click the **"🗑️ Clear Chat History"** button in the sidebar to start a new conversation.

## 🔧 Configuration

### Model Configuration (`config.py`)

Customize model settings, prompt formats, and generation parameters in `config.py`.

### Prompt Templates

The app uses a default prompt template. If your models were fine-tuned with a specific format (Alpaca, ChatML, etc.), update the `formatted_prompt` in the `generate_response()` function.

### Generation Parameters

Adjustable via the UI:
- **Temperature** (0.1-1.0): Controls randomness. Lower = more focused, Higher = more creative
- **Top P** (0.1-1.0): Nucleus sampling threshold
- **Max Tokens** (64-512): Maximum length of generated response

## 💾 Memory Requirements

Approximate VRAM usage with 4-bit quantization:
- **Mistral v3 7B**: ~4-5GB
- **Llama 3.1 8B**: ~5-6GB
- **Phi-4**: ~3-4GB

**Running all three models simultaneously**: ~12-15GB VRAM

**CPU-only mode**: Possible but significantly slower. Models will still load, but inference will take much longer.

## 🐛 Troubleshooting

### Model Loading Errors

```python
Error loading model: ...
```

**Solutions:**
- Verify your HuggingFace model paths are correct
- Ensure you have access to the models (if private)
- Check if you're logged in: `huggingface-cli whoami`
- Verify sufficient disk space for model downloads

### CUDA Out of Memory

```python
torch.cuda.OutOfMemoryError: ...
```

**Solutions:**
- Reduce max_tokens parameter
- Use only one model at a time (disable comparison mode)
- Close other GPU-intensive applications
- Restart the Streamlit app

### Slow Generation

**Solutions:**
- Ensure CUDA is available: Check "Active Model" in sidebar shows GPU usage
- Reduce max_tokens
- Consider upgrading GPU or using a cloud service

## 📁 Project Structure

```
Mental-Health-Agent/
├── app.py              # Main Streamlit application
├── config.py           # Configuration file for models
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## 🔐 Security Notes

- Never commit HuggingFace tokens to version control
- Use environment variables for sensitive data
- Keep your models private if they contain sensitive training data

## 🚀 Deployment Options

### Local Development
Follow the installation instructions above.

### Cloud Deployment

**Streamlit Cloud:**
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Add HuggingFace token as a secret
4. Deploy (Note: GPU instances may not be available)

**HuggingFace Spaces:**
1. Create a new Space with Streamlit
2. Upload your files
3. Set HF_TOKEN secret
4. Models will auto-download

**AWS/GCP/Azure:**
- Deploy on GPU-enabled instances
- Use container services (Docker)
- Configure security groups and access controls

## 📊 Model Comparison Insights

When using comparison mode, observe:
- **Response Length**: Which model provides more detailed responses?
- **Empathy Level**: Which model demonstrates better emotional understanding?
- **Safety**: Which model better avoids medical advice?
- **Coherence**: Which model maintains context better?

## 🤝 Contributing

This is a project submission, but improvements are welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 Citation

If you use this code or approach in your work, please cite:

```
Mental Health Chatbot Multi-Model Deployment
Fine-tuned models: Mistral v3 7B, Llama 3.1 8B, Phi-4 Conversational
Dataset: Mental Health Chatbot Pairs (Kaggle)
Technique: LoRA/QLoRA Parameter-Efficient Fine-Tuning
```

## 📄 License

This project is for educational purposes. Please ensure you comply with the licenses of:
- Base models (Mistral, Llama, Phi)
- Training dataset
- Third-party libraries

## 🙏 Acknowledgments

- **Unsloth** for efficient fine-tuning
- **HuggingFace** for model hosting and transformers library
- **Streamlit** for the web framework
- **Mental Health Chatbot Pairs** dataset creators

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review HuggingFace model pages for model-specific issues
3. Check Streamlit documentation for UI issues

---

**Built with ❤️ for mental health awareness and AI-powered support systems**
