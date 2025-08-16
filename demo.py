#!/usr/bin/env python3
"""
Fin-o1-8B Demo - Financial Reasoning Model
A fine-tuned version of Qwen3-8B for financial reasoning tasks
"""

import os
import sys
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

class FinO1Model:
    def __init__(self, model_path="./fin-o1-8b"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the Fin-o1-8B model and tokenizer"""
        try:
            print(f"🔄 Loading model from {self.model_path}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_response(self, prompt, max_length=512, temperature=0.7):
        """Generate response from the model"""
        if self.model is None or self.tokenizer is None:
            return "❌ Model not loaded. Please load the model first."
        
        try:
            # Prepare input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"❌ Error generating response: {e}"

def create_gradio_interface():
    """Create Gradio web interface"""
    model = FinO1Model()
    
    def chat_with_model(message, history, max_length, temperature):
        if not model.model:
            if not model.load_model():
                return "❌ Failed to load model. Please check if the model files exist.", history
        
        response = model.generate_response(message, max_length, temperature)
        history.append((message, response))
        return "", history
    
    def load_model_click():
        if model.load_model():
            return "✅ Model loaded successfully!"
        else:
            return "❌ Failed to load model. Please check if the model files exist."
    
    # Create Gradio interface
    with gr.Blocks(title="Fin-o1-8B Financial Reasoning Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🦙 Fin-o1-8B Financial Reasoning Demo
        
        **Fin-o1-8B** is a fine-tuned version of **Qwen3-8B**, designed to improve performance on financial reasoning tasks.
        
        This model has been trained using SFT and RF on financial datasets including FinQA, TATQA, DocMath-Eval, Econ-Logic, BizBench-QA, and DocFinQA.
        
        ## 📊 Model Details
        - **Base Model**: Qwen3-8B
        - **Parameters**: 8.19B
        - **Specialization**: Financial reasoning and mathematical tasks
        - **License**: Apache 2.0
        
        ## 🚀 Getting Started
        1. Click "Load Model" to initialize the model
        2. Ask financial or mathematical questions
        3. Adjust generation parameters as needed
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Model Controls")
                load_btn = gr.Button("🔄 Load Model", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("### 🎛️ Generation Parameters")
                max_length = gr.Slider(minimum=64, maximum=1024, value=512, step=64, 
                                     label="Max New Tokens")
                temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                                      label="Temperature")
                
                gr.Markdown("### 📚 Example Questions")
                examples = [
                    "What is the result of 3-5?",
                    "If a company has revenue of $1M and costs of $600K, what is the profit margin?",
                    "Calculate the compound interest on $1000 at 5% for 3 years.",
                    "What is the difference between simple and compound interest?",
                    "If a stock price increases from $50 to $75, what is the percentage gain?"
                ]
                gr.Examples(examples=examples, label="Try these examples:")
                
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat Interface")
                chatbot = gr.Chatbot(height=500, label="Fin-o1-8B")
                msg = gr.Textbox(label="Your Question", placeholder="Ask a financial or mathematical question...")
                clear = gr.Button("🗑️ Clear Chat")
        
        # Event handlers
        load_btn.click(load_model_click, outputs=load_status)
        msg.submit(chat_with_model, [msg, chatbot, max_length, temperature], [msg, chatbot])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        gr.Markdown("""
        ## 🔬 Research Citation
        If you use this model in your research, please cite:
        
        ```bibtex
        @article{qian2025fino1,
          title={Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance},
          author={Qian, Lingfei and Zhou, Weipeng and Wang, Yan and Peng, Xueqing and Huang, Jimin and Xie, Qianqian},
          journal={arXiv preprint arXiv:2502.08127},
          year={2025}
        }
        ```
        """)
    
    return demo

def command_line_demo():
    """Command line demo interface"""
    model = FinO1Model()
    
    print("🦙 Fin-o1-8B Financial Reasoning Demo")
    print("=" * 50)
    
    # Load model
    if not model.load_model():
        print("❌ Failed to load model. Please run 'python download_model.py' first.")
        return
    
    print("\n💬 Chat with the model (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n🤔 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🔄 Generating response...")
            response = model.generate_response(user_input)
            print(f"🤖 Fin-o1-8B: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        command_line_demo()
    else:
        # Launch Gradio interface
        demo = create_gradio_interface()
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)