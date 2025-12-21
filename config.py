# Mental Health Chatbot Configuration

# Update these with your actual HuggingFace model paths
MODEL_CONFIGS = {
    "mistral_v3_7b": {
        "model_id": "prranjal/16-bit-Mistral-7B",
        "display_name": "Mistral v3 7B",
        "description": "Mistral v3 7B fine-tuned for mental health support",
        "max_length": 512,
        "recommended_temp": 0.7,
        "recommended_top_p": 0.9
    },
    "llama_3_1_8b": {
        "model_id": "prranjal/16-bit-Llama3.1-8B",
        "display_name": "Llama 3.1 8B",
        "description": "Llama 3.1 8B fine-tuned for mental health support",
        "max_length": 512,
        "recommended_temp": 0.7,
        "recommended_top_p": 0.9
    },
    "phi_4_conversational": {
        "model_id": "prranjal/16-bit-Phi_4-Conversational",
        "display_name": "Phi-4 Conversational",
        "description": "Phi-4 fine-tuned for mental health conversations",
        "max_length": 512,
        "recommended_temp": 0.7,
        "recommended_top_p": 0.9
    }
}

# Generation parameters
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 256,
    "repetition_penalty": 1.1,
    "do_sample": True
}

# Quantization settings
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True
}

# System prompts (customize based on your training format)
SYSTEM_PROMPTS = {
    "default": """Below is a conversation with a mental health support assistant. The assistant is empathetic, supportive, and non-judgmental.

User: {user_input}
Assistant:""",
    
    "alpaca": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are a mental health support assistant. Provide empathetic, supportive, and non-judgmental responses.

### Input:
{user_input}

### Response:""",
    
    "chatml": """<|im_start|>system
You are a mental health support assistant. You are empathetic, supportive, and non-judgmental.<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
}

# Use this prompt format (change based on how you fine-tuned your models)
ACTIVE_PROMPT_FORMAT = "default"
