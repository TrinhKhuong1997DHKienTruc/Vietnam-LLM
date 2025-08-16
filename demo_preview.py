#!/usr/bin/env python3
"""
Fin-o1-8B Demo Preview
Shows the interface without requiring model download
"""

import gradio as gr

def create_preview_interface():
    """Create a preview interface showing the demo capabilities"""
    
    def mock_response(message, history, max_length, temperature):
        """Mock response to demonstrate the interface"""
        if not message.strip():
            return "", history
        
        # Simulate model thinking
        import time
        time.sleep(1)
        
        # Mock responses based on input
        mock_responses = {
            "what is 2+2": "The result of 2+2 is 4. This is a basic arithmetic operation where we add two numbers together.",
            "calculate profit margin": "To calculate profit margin, you need to know the revenue and costs. The formula is: Profit Margin = (Revenue - Costs) / Revenue × 100%. For example, if revenue is $1000 and costs are $600, the profit margin would be (1000-600)/1000 × 100% = 40%.",
            "compound interest": "Compound interest is interest earned on both the principal amount and any accumulated interest. The formula is: A = P(1 + r/n)^(nt), where A is the final amount, P is principal, r is annual interest rate, n is number of times interest is compounded per year, and t is time in years.",
            "stock price": "Stock price changes are calculated as: ((New Price - Old Price) / Old Price) × 100%. For example, if a stock goes from $50 to $75, the percentage gain is ((75-50)/50) × 100% = 50%.",
            "financial ratio": "Common financial ratios include: 1) Current Ratio = Current Assets / Current Liabilities, 2) Debt-to-Equity = Total Debt / Total Equity, 3) Return on Equity = Net Income / Shareholders' Equity, 4) Gross Margin = (Revenue - COGS) / Revenue."
        }
        
        # Find best matching mock response
        message_lower = message.lower()
        response = "I'm a financial reasoning AI model. I can help with calculations, financial analysis, and mathematical problems. This is a preview - the actual model would provide more detailed and accurate responses."
        
        for key, value in mock_responses.items():
            if key in message_lower:
                response = value
                break
        
        history.append((message, response))
        return "", history
    
    def load_model_click():
        return "🔄 This is a preview mode. To use the real model, download it first with 'python download_model.py'"
    
    # Create Gradio interface
    with gr.Blocks(title="Fin-o1-8B Demo Preview", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🦙 Fin-o1-8B Financial Reasoning Demo - PREVIEW MODE
        
        **This is a preview of the Fin-o1-8B interface. The actual model is not loaded.**
        
        **Fin-o1-8B** is a fine-tuned version of **Qwen3-8B**, designed to excel at financial reasoning tasks.
        
        ## 📊 Model Details
        - **Base Model**: Qwen3-8B
        - **Parameters**: 8.19B
        - **Specialization**: Financial reasoning and mathematical tasks
        - **License**: Apache 2.0
        
        ## 🚀 To Use the Real Model
        1. Download the model: `python download_model.py`
        2. Run the full demo: `python demo.py`
        3. Or use command line: `python demo.py --cli`
        
        ## 💡 Try These Example Questions
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Model Controls")
                load_btn = gr.Button("🔄 Load Model", variant="primary")
                load_status = gr.Textbox(label="Status", interactive=False, value="Preview mode - model not loaded")
                
                gr.Markdown("### 🎛️ Generation Parameters")
                max_length = gr.Slider(minimum=64, maximum=1024, value=512, step=64, 
                                     label="Max New Tokens")
                temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                                      label="Temperature")
                
                gr.Markdown("### 📚 Example Questions")
                examples = [
                    "What is 2+2?",
                    "Calculate profit margin",
                    "Explain compound interest",
                    "Calculate stock price change",
                    "What are financial ratios?"
                ]
                gr.Examples(examples=examples, label="Try these examples:")
                
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat Interface (Preview Mode)")
                chatbot = gr.Chatbot(height=500, label="Fin-o1-8B Preview")
                msg = gr.Textbox(label="Your Question", placeholder="Ask a financial or mathematical question...")
                clear = gr.Button("🗑️ Clear Chat")
                
                gr.Markdown("""
                **Note**: This preview shows mock responses. The actual model will provide real, 
                detailed financial reasoning based on its training data.
                """)
        
        # Event handlers
        load_btn.click(load_model_click, outputs=load_status)
        msg.submit(mock_response, [msg, chatbot, max_length, temperature], [msg, chatbot])
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
        
        ## 📚 Next Steps
        - **Download the model**: `python download_model.py`
        - **Run full demo**: `python demo.py`
        - **Read documentation**: Check README.md and other guides
        - **Use Docker**: `docker build -t fin-o1-8b . && docker run -p 7860:7860 fin-o1-8b`
        """)
    
    return demo

if __name__ == "__main__":
    print("🦙 Fin-o1-8B Demo Preview")
    print("This shows the interface without downloading the model.")
    print("To use the real model, run: python download_model.py")
    print("")
    
    # Launch preview interface
    demo = create_preview_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)