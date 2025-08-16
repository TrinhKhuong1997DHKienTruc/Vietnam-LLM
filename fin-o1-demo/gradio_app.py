#!/usr/bin/env python3
"""
Gradio Web App for Fin-o1-8B
A simple and clean Gradio interface for the Fin-o1-8B financial reasoning model.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import warnings
warnings.filterwarnings("ignore")

class FinO1Gradio:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the Fin-o1-8B model."""
        if self.is_loaded:
            return "✅ Model already loaded!"
        
        try:
            with gr.Blocks() as loading_block:
                gr.Markdown("🚀 Loading Fin-o1-8B model... Please wait.")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "TheFinAI/Fin-o1-8B",
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                "TheFinAI/Fin-o1-8B",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.is_loaded = True
            return "✅ Model loaded successfully!"
            
        except Exception as e:
            return f"❌ Error loading model: {e}"
    
    def generate_response(self, prompt, max_tokens, temperature):
        """Generate response from the model."""
        if not self.is_loaded:
            return "❌ Please load the model first!"
        
        if not prompt.strip():
            return "❌ Please enter a prompt!"
        
        try:
            start_time = time.time()
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            return f"🤖 **Response:**\n\n{response}\n\n⏱️ Generated in {generation_time:.2f} seconds"
            
        except Exception as e:
            return f"❌ Error generating response: {e}"
    
    def run_examples(self):
        """Run predefined financial examples."""
        examples = [
            "What is the result of 3-5?",
            "If I invest $1000 at 5% annual interest for 3 years, how much will I have?",
            "A company has revenue of $1M and expenses of $800K. What is the profit margin percentage?",
            "Explain the relationship between inflation and interest rates.",
            "What are the key factors to consider when evaluating a company's financial health?"
        ]
        
        results = []
        for i, example in enumerate(examples, 1):
            result = self.generate_response(example, 512, 0.7)
            results.append(f"**Example {i}:** {example}\n\n{result}\n\n---\n")
        
        return "\n".join(results)

def create_interface():
    """Create the Gradio interface."""
    
    # Initialize the model handler
    model_handler = FinO1Gradio()
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(css=css, title="Fin-o1-8B Financial AI") as interface:
        
        # Header
        gr.Markdown("""
        # 💰 Fin-o1-8B Financial AI
        
        **Fin-o1-8B** is a fine-tuned version of **Qwen3-8B**, designed for financial reasoning tasks.
        
        **Base Model**: Qwen3-8B | **Training**: SFT + RF | **Paper**: [arXiv:2502.08127](https://arxiv.org/abs/2502.08127)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model loading section
                gr.Markdown("### 🚀 Model Setup")
                load_btn = gr.Button("📥 Load Model", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False)
                
                # Model parameters
                gr.Markdown("### ⚙️ Generation Parameters")
                max_tokens = gr.Slider(100, 1000, 512, step=50, label="Max Tokens")
                temperature = gr.Slider(0.1, 1.5, 0.7, step=0.1, label="Temperature")
                
                # Device info
                device_info = "🎮 GPU" if torch.cuda.is_available() else "🖥️ CPU"
                gr.Markdown(f"### {device_info}")
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gr.Markdown(f"**GPU:** {gpu_name}")
                else:
                    gr.Markdown("**Device:** CPU")
            
            with gr.Column(scale=2):
                # Main chat interface
                gr.Markdown("### 💬 Financial AI Chat")
                prompt_input = gr.Textbox(
                    label="Ask me about finance, math, or economics...",
                    placeholder="e.g., What is the result of 3-5?",
                    lines=3
                )
                generate_btn = gr.Button("🚀 Generate Response", variant="primary")
                response_output = gr.Markdown(label="Response")
                
                # Examples section
                gr.Markdown("### 📊 Quick Examples")
                examples_btn = gr.Button("🎯 Run All Examples")
                examples_output = gr.Markdown(label="Examples Results")
        
        # Event handlers
        load_btn.click(
            fn=model_handler.load_model,
            outputs=load_status
        )
        
        generate_btn.click(
            fn=model_handler.generate_response,
            inputs=[prompt_input, max_tokens, temperature],
            outputs=response_output
        )
        
        examples_btn.click(
            fn=model_handler.run_examples,
            outputs=examples_output
        )
        
        # Enter key support for prompt input
        prompt_input.submit(
            fn=model_handler.generate_response,
            inputs=[prompt_input, max_tokens, temperature],
            outputs=response_output
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )