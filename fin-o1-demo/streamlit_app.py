#!/usr/bin/env python3
"""
Streamlit Web App for Fin-o1-8B
A modern web interface for the Fin-o1-8B financial reasoning model.
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Fin-o1-8B Financial AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .response-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the Fin-o1-8B model with caching."""
    try:
        with st.spinner("🚀 Loading Fin-o1-8B model... This may take a few minutes on first run."):
            tokenizer = AutoTokenizer.from_pretrained(
                "TheFinAI/Fin-o1-8B",
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                "TheFinAI/Fin-o1-8B",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            st.success("✅ Model loaded successfully!")
            return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate response from the model."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response
        
    except Exception as e:
        return f"Error generating response: {e}"

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">💰 Fin-o1-8B Financial AI</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model parameters
    max_tokens = st.sidebar.slider("Max Tokens", 100, 1000, 512, 50)
    temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    
    # Device info
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"🖥️ Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.info(f"🎮 GPU: {gpu_name}")
    
    # Model information
    st.sidebar.markdown("""
    ### 📚 Model Info
    **Fin-o1-8B** is a fine-tuned version of **Qwen3-8B**, designed for financial reasoning tasks.
    
    **Base Model**: Qwen3-8B  
    **Training**: SFT and RF on financial datasets  
    **Paper**: [arXiv:2502.08127](https://arxiv.org/abs/2502.08127)
    """)
    
    # Load model
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load model. Please check the console for errors.")
        return
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Examples", "📈 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("💬 Financial AI Chat")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about finance, math, or economics..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    start_time = time.time()
                    response = generate_response(model, tokenizer, prompt, max_tokens, temperature)
                    end_time = time.time()
                    
                    st.markdown(response)
                    st.caption(f"Generated in {end_time - start_time:.2f} seconds")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    with tab2:
        st.header("📊 Financial Examples")
        
        # Predefined examples
        examples = {
            "Basic Math": "What is the result of 3-5?",
            "Compound Interest": "If I invest $1000 at 5% annual interest for 3 years, how much will I have?",
            "Profit Margin": "A company has revenue of $1M and expenses of $800K. What is the profit margin percentage?",
            "Inflation & Rates": "Explain the relationship between inflation and interest rates.",
            "Financial Health": "What are the key factors to consider when evaluating a company's financial health?",
            "ROI Calculation": "Calculate the ROI if I invest $5000 and get back $6500 after 2 years.",
            "Break-even Analysis": "If my fixed costs are $10,000 and variable cost per unit is $5, at what price should I sell to break even with 1000 units?",
            "Portfolio Diversification": "Explain the benefits of portfolio diversification with examples."
        }
        
        # Create columns for examples
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Example Questions")
            for category, question in list(examples.items())[:4]:
                if st.button(f"💡 {category}", key=f"ex1_{category}"):
                    st.session_state.current_example = question
                    st.rerun()
        
        with col2:
            st.subheader("📝 More Examples")
            for category, question in list(examples.items())[4:]:
                if st.button(f"💡 {category}", key=f"ex2_{category}"):
                    st.session_state.current_example = question
                    st.rerun()
        
        # Display current example and generate response
        if hasattr(st.session_state, 'current_example'):
            st.markdown("---")
            st.subheader("🎯 Current Example")
            st.markdown(f"**Question:** {st.session_state.current_example}")
            
            if st.button("🚀 Generate Answer"):
                with st.spinner("🤖 Generating response..."):
                    start_time = time.time()
                    response = generate_response(
                        model, tokenizer, 
                        st.session_state.current_example, 
                        max_tokens, temperature
                    )
                    end_time = time.time()
                
                st.markdown("**Answer:**")
                st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
                st.caption(f"Generated in {end_time - start_time:.2f} seconds")
    
    with tab3:
        st.header("📈 Analytics Dashboard")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Size", "8.19B Parameters")
        
        with col2:
            st.metric("Base Model", "Qwen3-8B")
        
        with col3:
            st.metric("Training Method", "SFT + RF")
        
        with col4:
            st.metric("License", "Apache 2.0")
        
        # Sample financial data visualization
        st.subheader("📊 Sample Financial Data")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023, 2024],
            'Revenue': [1000000, 1200000, 1100000, 1400000, 1600000],
            'Expenses': [800000, 900000, 950000, 1100000, 1200000],
            'Profit': [200000, 300000, 150000, 300000, 400000]
        })
        
        # Revenue vs Expenses chart
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=sample_data['Year'], y=sample_data['Revenue'], 
                                 mode='lines+markers', name='Revenue', line=dict(color='green')))
        fig1.add_trace(go.Scatter(x=sample_data['Year'], y=sample_data['Expenses'], 
                                 mode='lines+markers', name='Expenses', line=dict(color='red')))
        fig1.update_layout(title='Revenue vs Expenses Over Time', xaxis_title='Year', yaxis_title='Amount ($)')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Profit trend
        fig2 = px.bar(sample_data, x='Year', y='Profit', title='Profit Trend Over Time',
                     color='Profit', color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.header("ℹ️ About Fin-o1-8B")
        
        st.markdown("""
        ### 🧠 Model Overview
        
        **Fin-o1-8B** is a state-of-the-art language model specifically designed for financial reasoning tasks. 
        It's built upon the foundation of Qwen3-8B and fine-tuned using advanced techniques to excel in 
        financial mathematics, economic logic, and business analysis.
        
        ### 🔬 Technical Details
        
        - **Base Model**: Qwen3-8B (8.19B parameters)
        - **Training Method**: Supervised Fine-Tuning (SFT) + Reinforcement Learning (RF)
        - **Training Data**: FinCoT dataset derived from FinQA, TATQA, DocMath-Eval, Econ-Logic, BizBench-QA, DocFinQA
        - **Model Size**: 8.19B parameters
        - **License**: Apache 2.0
        
        ### 📚 Research Paper
        
        This model is based on the research paper:
        **"Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance"**
        
        **Authors**: Qian, Lingfei; Zhou, Weipeng; Wang, Yan; Peng, Xueqing; Huang, Jimin; Xie, Qianqian
        
        **arXiv**: [2502.08127](https://arxiv.org/abs/2502.08127)
        
        ### 🎯 Use Cases
        
        - Financial calculations and reasoning
        - Economic concept explanations
        - Business analysis and evaluation
        - Mathematical problem solving
        - Investment analysis
        - Risk assessment
        
        ### 🚀 Getting Started
        
        1. Use the **Chat** tab to ask financial questions
        2. Explore **Examples** for pre-defined financial scenarios
        3. Adjust parameters in the sidebar for different response styles
        4. View **Analytics** for sample financial visualizations
        
        ### 📞 Support
        
        For more information, visit the [Hugging Face model page](https://huggingface.co/TheFinAI/Fin-o1-8B).
        """)

if __name__ == "__main__":
    main()