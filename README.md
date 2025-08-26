## Fin-o1-8B (Hugging Face) - Quick Start (Windows/Linux)

Mã nguồn này chuẩn bị sẵn demo chạy model Fin-o1-8B từ Hugging Face (`TheFinAI/Fin-o1-8B`).

- **Python**: ưu tiên Python 3.13 (nếu một vài dependency chưa hỗ trợ, script sẽ tự fallback dùng Python 3.12 nếu có).
- **Runtime**: chạy được trên CPU (chậm, cần nhiều RAM), tự động dùng GPU nếu phát hiện.
- **Cài nhanh**: không cần Git, người dùng có thể ấn "Download ZIP" và chạy script.

### Cài đặt và chạy

1) Giải nén ZIP, mở Terminal (Windows PowerShell hoặc Linux bash) vào thư mục `fin-o1/`.

2) Windows:

```bat
cd fin-o1
run_demo.bat --prompt "Tính 3 - 5 là bao nhiêu?"
```

   Linux:

```bash
cd fin-o1
chmod +x run_demo.sh
./run_demo.sh --prompt "Tính 3 - 5 là bao nhiêu?"
```

Lần chạy đầu sẽ tự tạo `venv`, cài `requirements.txt`, và tải model từ Hugging Face.

### Yêu cầu phần cứng và lưu ý

- Chạy trên **CPU** có thể cần ≥ 32GB RAM và sẽ chậm. Khuyến nghị dùng **GPU NVIDIA** nếu có.
- Nếu có GPU và `bitsandbytes` tương thích, script sẽ tự bật **4-bit quantization** để tiết kiệm VRAM.
- Mặc định `requirements.txt` dùng gói `torch` từ CPU index. Người dùng GPU có thể chủ động cài bản GPU của PyTorch theo hướng dẫn chính thức: `pytorch.org`.

### Demo script

File `fin-o1/fin_o1_demo.py` sẽ:

- Tự phát hiện GPU/CPU, chọn dtype phù hợp (bf16/fp16 trên GPU, fp32 trên CPU).
- Nếu có, áp dụng **chat template** của tokenizer để prompt.
- Sinh tối đa 64 token (có thể chỉnh bằng `--max-new-tokens`).

Ví dụ lệnh:

```bash
python fin_o1_demo.py --prompt "Hãy giải thích ngắn gọn: 3 - 5 bằng bao nhiêu?" --max-new-tokens 64
```

### Thư mục liên quan

- `fin-o1/requirements.txt`: dependencies tương thích Python 3.13 (có fallback 3.12 trong script chạy).
- `fin-o1/run_demo.bat`, `fin-o1/run_demo.sh`: script chạy nhanh cho Windows/Linux.
- `.gitignore`: bỏ qua `venv`, cache, v.v.

### Ghi chú

- Nếu gặp lỗi khi cài dependencies trên Python 3.13 (do wheel chưa phát hành), hãy chạy lại script trên Python 3.12.
- Model sẽ được tải từ `huggingface.co/TheFinAI/Fin-o1-8B` khi chạy lần đầu (cần Internet).
