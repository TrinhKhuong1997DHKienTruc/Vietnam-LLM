#!/usr/bin/env python
import os
from typing import Optional

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def create_generator(model_id: str, device_preference: Optional[str] = None):
	os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
	trust_remote_code = True

	if device_preference is None:
		if torch.cuda.is_available():
			device_preference = "cuda"
		elif torch.backends.mps.is_available():
			device_preference = "mps"
		else:
			device_preference = "cpu"

	if device_preference == "cuda":
		dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
		device_map = "auto"
	elif device_preference == "mps":
		dtype = torch.float16
		device_map = {"": "mps"}
	else:
		dtype = torch.float32
		device_map = {"": "cpu"}

	tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
	model = AutoModelForCausalLM.from_pretrained(
		model_id,
		trust_remote_code=trust_remote_code,
		torch_dtype=dtype,
		device_map=device_map,
	)

	gen = pipeline(
		"text-generation",
		model=model,
		tokenizer=tokenizer,
		return_full_text=False,
		batch_size=1,
	)
	return gen


generator_cache = {"pipeline": None, "model_id": None, "device": None}


def generate(prompt, max_new_tokens, temperature, top_p, model_id, device):
	if generator_cache["pipeline"] is None or generator_cache["model_id"] != model_id or generator_cache["device"] != device:
		generator_cache["pipeline"] = create_generator(model_id, None if device == "auto" else device)
		generator_cache["model_id"] = model_id
		generator_cache["device"] = device

	outputs = generator_cache["pipeline"](
		prompt,
		max_new_tokens=int(max_new_tokens),
		do_sample=True,
		temperature=float(temperature),
		top_p=float(top_p),
		repetition_penalty=1.05,
	)
	return outputs[0]["generated_text"]


def app():
	with gr.Blocks(title="Fin-o1-8B Demo") as demo:
		gr.Markdown("# Fin-o1-8B Demo\nEnter a prompt and generate text.")
		with gr.Row():
			prompt = gr.Textbox(label="Prompt", value="List 3 short key takeaways from AAPL's latest earnings.", lines=4)
		with gr.Row():
			max_new_tokens = gr.Slider(16, 1024, value=256, step=8, label="max_new_tokens")
			temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="temperature")
			top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
		with gr.Row():
			model_id = gr.Textbox(label="Model ID", value="TheFinAI/Fin-o1-8B")
			device = gr.Dropdown(["auto", "cpu", "cuda", "mps"], value="auto", label="Device")
		btn = gr.Button("Generate")
		output = gr.Textbox(label="Output", lines=12)

		btn.click(generate, inputs=[prompt, max_new_tokens, temperature, top_p, model_id, device], outputs=output)
	return demo


if __name__ == "__main__":
	app().launch()