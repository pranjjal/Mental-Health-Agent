import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from datetime import datetime
import traceback
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #4A90E2;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
        margin-left: 20%;
        color: #1a1a1a;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
        color: #1a1a1a;
    }
    .model-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Model configurations
MODEL_CONFIGS = {
    "Mistral v3 7B": {
        "model_id": "prranjal/16-bit-Mistral-7B", 
        "description": "Mistral v3 7B fine-tuned for mental health support",
        "max_length": 512,
        "use_api": True,
        "api_endpoint": "https://roastingly-diathetic-ema.ngrok-free.dev/submit_mistral"
    },
    "Llama 3.1 8B": {
        "model_id": "prranjal/16-bit-Llama3.1-8B",
        "description": "Llama 3.1 8B fine-tuned for mental health support",
        "max_length": 512,
        "use_api": True,
        "api_endpoint": "https://roastingly-diathetic-ema.ngrok-free.dev/submit_llama"
    },
    "Phi-4 Conversational": {
        "model_id": "prranjal/16-bit-Phi_4-Conversational",
        "description": "Phi-4 fine-tuned for mental health conversations",
        "max_length": 512,
        "use_api": False
    }
}

# Load model and tokenizer
@st.cache_resource
def load_model(model_name):
    """Load model and tokenizer with caching"""
    try:
        config = MODEL_CONFIGS[model_name]
        model_id = config["model_id"]
        
        with st.spinner(f"Loading {model_name}... This may take a few minutes on first load."):
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                trust_remote_code=True,
                use_fast=False
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Try loading with different strategies
            model = None
            
            # Strategy 1: Load in 16-bit as-is (since your models are already 16-bit)
            try:
                st.info("Loading model in 16-bit (FP16)...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            except Exception as e1:
                st.warning(f"16-bit loading failed, trying 8-bit quantization... Error: {str(e1)[:100]}")
                
                # Strategy 2: Try 8-bit quantization
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="auto",
                        trust_remote_code=True,
                        load_in_8bit=True,
                        low_cpu_mem_usage=True
                    )
                except Exception as e2:
                    st.warning(f"8-bit loading failed, trying full precision... Error: {str(e2)[:100]}")
                    
                    # Strategy 3: Try full precision (slower but most compatible)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
            
            if model is not None:
                model.eval()
                st.success(f"✓ {model_name} loaded successfully!")
            
        return model, tokenizer, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

def generate_response(model, tokenizer, prompt, config, temperature=0.7, top_p=0.9, max_new_tokens=256):
    """Generate response from the model - supports both local and API-based models"""
    try:
        # Check if this model uses API
        if config.get("use_api", False):
            return generate_response_api(prompt, config)
        
        # Local model generation
        # Format prompt for mental health context
        formatted_prompt = f"""Below is a conversation with a mental health support assistant. The assistant is empathetic, supportive, and non-judgmental.

User: {prompt}
Assistant:"""
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=config["max_length"])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_response_api(prompt, config):
    """Generate response using API endpoint"""
    try:
        api_endpoint = config.get("api_endpoint")
        
        if not api_endpoint:
            return "Error: API endpoint not configured"
        
        # Make POST request to API with JSON body
        headers = {
            "ngrok-skip-browser-warning": "true",
            "Content-Type": "application/json"
        }
        
        # Send prompt as JSON body
        response = requests.post(
            api_endpoint,
            json={"prompt": prompt},
            headers=headers,
            timeout=120  # Increased timeout for model processing
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response from API")
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model might be taking too long to respond."
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to API. Please check if the ngrok tunnel is active."
    except Exception as e:
        return f"Error calling API: {str(e)}"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "model_instance" not in st.session_state:
    st.session_state.model_instance = None
if "tokenizer_instance" not in st.session_state:
    st.session_state.tokenizer_instance = None
if "model_config" not in st.session_state:
    st.session_state.model_config = None

# Main UI
st.markdown('<div class="main-header">🧠 Mental Health Support Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Fine-tuned LLMs for Empathetic Mental Health Conversations</div>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important Disclaimer:</strong><br>
        This chatbot is for informational and emotional support purposes only. It does not provide medical diagnosis or treatment. 
        If you're experiencing a mental health crisis, please contact a professional healthcare provider or emergency services immediately.
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        options=list(MODEL_CONFIGS.keys()),
        help="Choose which fine-tuned model to use for the conversation"
    )
    
    # Display model info
    if selected_model:
        st.info(f"**{selected_model}**\n\n{MODEL_CONFIGS[selected_model]['description']}")
    
    st.divider()
    
    # Generation parameters
    st.subheader("Generation Parameters")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1, help="Higher values make output more random")
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.1, help="Nucleus sampling threshold")
    max_tokens = st.slider("Max Tokens", 64, 512, 256, 32, help="Maximum length of generated response")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Comparison mode
    st.subheader("🔄 Comparison Mode")
    compare_mode = st.checkbox("Enable Model Comparison", help="Generate responses from all three models simultaneously")
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Messages", len(st.session_state.messages))
    if st.session_state.current_model:
        st.metric("Active Model", st.session_state.current_model)

# Load model if changed
if selected_model != st.session_state.current_model:
    st.session_state.current_model = selected_model
    
    # Check if this model uses API (skip loading if it does)
    config_check = MODEL_CONFIGS[selected_model]
    if config_check.get("use_api", False):
        st.info(f"✓ {selected_model} configured to use API endpoint")
        st.session_state.model_instance = None
        st.session_state.tokenizer_instance = None
        st.session_state.model_config = config_check
    else:
        # Load model locally
        model, tokenizer, config = load_model(selected_model)
        st.session_state.model_instance = model
        st.session_state.tokenizer_instance = tokenizer
        st.session_state.model_config = config

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 {message.get('model', 'Assistant')}:</strong><br>
                {message["content"]}
                <div class="model-info">{message.get('timestamp', '')}</div>
            </div>
        """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Share what's on your mind... I'm here to listen.")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {user_input}
        </div>
    """, unsafe_allow_html=True)
    
    if compare_mode:
        # Generate responses from all models
        st.markdown("### 📊 Model Comparison")
        
        cols = st.columns(3)
        
        for idx, (model_name, model_config) in enumerate(MODEL_CONFIGS.items()):
            with cols[idx]:
                st.subheader(model_name)
                with st.spinner(f"Generating response from {model_name}..."):
                    # Load model if not current
                    if model_name != st.session_state.current_model:
                        temp_model, temp_tokenizer, temp_config = load_model(model_name)
                    else:
                        temp_model = st.session_state.model_instance
                        temp_tokenizer = st.session_state.tokenizer_instance
                        temp_config = st.session_state.model_config
                    
                    if temp_model and temp_tokenizer:
                        response = generate_response(
                            temp_model, 
                            temp_tokenizer, 
                            user_input, 
                            temp_config,
                            temperature, 
                            top_p, 
                            max_tokens
                        )
                        
                        st.markdown(f"""
                            <div class="chat-message bot-message">
                                {response}
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to load model")
        
        # Add comparison note to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Comparison mode: Responses generated from all three models above.",
            "model": "Comparison",
            "timestamp": timestamp
        })
        
    else:
        # Generate single response
        if st.session_state.model_config:
            # Check if using API or local model
            if st.session_state.model_config.get("use_api", False):
                # API-based generation
                with st.spinner(f"Generating response from {selected_model} (via API)..."):
                    response = generate_response_api(
                        user_input,
                        st.session_state.model_config
                    )
            elif st.session_state.model_instance and st.session_state.tokenizer_instance:
                # Local model generation
                with st.spinner(f"Generating response from {selected_model}..."):
                    response = generate_response(
                        st.session_state.model_instance,
                        st.session_state.tokenizer_instance,
                        user_input,
                        st.session_state.model_config,
                        temperature,
                        top_p,
                        max_tokens
                    )
            else:
                response = "Error: Model not loaded properly."
                
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "model": selected_model,
                "timestamp": timestamp
            })
            
            # Display assistant message
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 {selected_model}:</strong><br>
                    {response}
                    <div class="model-info">{timestamp}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Model not configured. Please select a model from the sidebar.")
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p>💙 Remember: You're not alone. There's always help available.</p>
        <p><strong>Crisis Resources:</strong> National Suicide Prevention Lifeline: 988 | Crisis Text Line: Text HOME to 741741</p>
    </div>
""", unsafe_allow_html=True)
