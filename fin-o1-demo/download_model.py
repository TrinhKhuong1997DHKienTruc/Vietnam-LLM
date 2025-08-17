#!/usr/bin/env python
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
	parser = argparse.ArgumentParser(description="Pre-download a Hugging Face model snapshot")
	parser.add_argument("--model-id", default="TheFinAI/Fin-o1-8B")
	parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="Optional HF token (or set HF_TOKEN env)")
	parser.add_argument("--local-dir", default=str(Path("models") / "TheFinAI__Fin-o1-8B"))
	args = parser.parse_args()

	os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
	local_dir = Path(args.local_dir)
	local_dir.mkdir(parents=True, exist_ok=True)

	print(f"Downloading {args.model_id} to {local_dir} ...")
	path = snapshot_download(
		args.model_id,
		local_dir=str(local_dir),
		local_dir_use_symlinks=False,
		use_auth_token=(args.token if args.token else None),
	)
	print(f"Done. Files in: {path}")


if __name__ == "__main__":
	main()