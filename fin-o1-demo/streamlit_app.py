#!/usr/bin/env python
import os
from typing import Optional

import torch
import streamlit as st
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


@st.cache_resource(show_spinner=False)
def get_generator(model_id: str, device: str):
	device_pref = None if device == "auto" else device
	return create_generator(model_id, device_pref)


def main():
	st.set_page_config(page_title="Fin-o1-8B Demo", layout="wide")
	st.title("Fin-o1-8B Demo")

	with st.sidebar:
		st.header("Settings")
		model_id = st.text_input("Model ID", value="TheFinAI/Fin-o1-8B")
		device = st.selectbox("Device", options=["auto", "cpu", "cuda", "mps"], index=0)
		max_new_tokens = st.slider("max_new_tokens", 16, 1024, 256, 8)
		temperature = st.slider("temperature", 0.0, 2.0, 0.7, 0.05)
		top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05)

	prompt = st.text_area("Prompt", value="List 3 short key takeaways from AAPL's latest earnings.", height=160)

	if st.button("Generate", use_container_width=True):
		with st.spinner("Loading model and generating..."):
			generator = get_generator(model_id, device)
			outputs = generator(
				prompt,
				max_new_tokens=int(max_new_tokens),
				do_sample=True,
				temperature=float(temperature),
				top_p=float(top_p),
				repetition_penalty=1.05,
			)
			text = outputs[0]["generated_text"]
		st.subheader("Output")
		st.write(text)


if __name__ == "__main__":
	main()