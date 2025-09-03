# FinRobot - AI Agent Platform for Financial Analysis

<div align="center">
<img align="center" width="30%" alt="FinRobot Logo" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/e0371951-1ce1-488e-aa25-0992dafcc139">
</div>

## 📊 Overview

**FinRobot** là một nền tảng AI Agent toàn diện được thiết kế đặc biệt cho các ứng dụng tài chính. Nó tích hợp **nhiều công nghệ AI khác nhau**, vượt ra ngoài phạm vi của các mô hình ngôn ngữ đơn thuần. Tầm nhìn mở rộng này làm nổi bật tính linh hoạt và khả năng thích ứng của nền tảng, đáp ứng các nhu cầu đa dạng của ngành tài chính.

**Khái niệm AI Agent**: AI Agent là một thực thể thông minh sử dụng các mô hình ngôn ngữ lớn làm não bộ để nhận thức môi trường, đưa ra quyết định và thực hiện các hành động. Khác với trí tuệ nhân tạo truyền thống, AI Agents có khả năng suy nghĩ độc lập và sử dụng các công cụ để đạt được các mục tiêu đã định một cách tiến bộ.

## 🚀 Tính năng chính

### 1. Market Forecaster Agent (Dự báo xu hướng cổ phiếu)
- Phân tích thông tin công ty và tin tức thị trường
- Dự báo xu hướng giá cổ phiếu trong tuần tới
- Cung cấp phân tích chi tiết về các yếu tố tích cực và tiêu cực

### 2. Financial Analyst Agent (Phân tích báo cáo tài chính)
- Viết báo cáo nghiên cứu cổ phiếu dựa trên dữ liệu 10-K
- Phân tích báo cáo tài chính và dữ liệu thị trường
- Tạo báo cáo PDF tự động

### 3. Trade Strategist Agent (Chiến lược giao dịch)
- Phát triển chiến lược giao dịch đa phương thức
- Tích hợp phân tích kỹ thuật và cơ bản

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Windows/Linux/macOS
- Kết nối internet để truy cập API

### Bước 1: Tải xuống và cài đặt
```bash
# Clone repository
git clone https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM.git

# Chuyển vào thư mục FinRobot
cd Vietnam-LLM/FinRobot

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt package trong chế độ development
pip install -e .
```

### Bước 2: Cấu hình API Keys

#### 1. Cấu hình OpenAI API
Chỉnh sửa file `OAI_CONFIG_LIST`:
```json
[
    {
        "model": "gpt-5-mini-2025-08-07",
        "api_key": "<your API key here>",
        "base_url": "<your base URL here>"
    }
]
```

#### 2. Cấu hình Financial APIs
Chỉnh sửa file `config_api_keys`:
```json
{
    "FINNHUB_API_KEY": "<your FINNHUB API key here>",
    "FMP_API_KEY": "<your FMP API key here>",
    "SEC_API_KEY": "<your SEC API key here>"
}
```

## 📈 Sử dụng

### Chạy Market Forecaster Demo

#### Phương pháp 1: Sử dụng script Python
```bash
# Chạy demo dự báo cho MSFT và NVDA
python mock_forecaster_demo.py
```

#### Phương pháp 2: Sử dụng Jupyter Notebook
```bash
# Khởi động Jupyter
jupyter notebook

# Mở file: tutorials_beginner/agent_fingpt_forecaster.ipynb
```

### Kết quả mẫu

#### Microsoft Corporation (MSFT)
- **Dự báo**: UP 2.5%
- **Yếu tố tích cực**:
  - Tăng trưởng mạnh trong cloud computing với Azure
  - Tích hợp AI trên toàn bộ danh mục sản phẩm
  - Nhu cầu phần mềm doanh nghiệp mạnh mẽ
- **Mối quan ngại**:
  - Giám sát quy định về thực hành AI
  - Cạnh tranh từ các startup AI mới nổi

#### NVIDIA Corporation (NVDA)
- **Dự báo**: UP 3.2%
- **Yếu tố tích cực**:
  - Vị thế thống trị trong thị trường chip AI
  - Nhu cầu mạnh mẽ cho GPU data center
  - Đổi mới trong phần cứng AI
- **Mối quan ngại**:
  - Ràng buộc chuỗi cung ứng
  - Căng thẳng địa chính trị ảnh hưởng xuất khẩu

## 🏗️ Kiến trúc hệ thống

### Bốn lớp chính của FinRobot:

1. **Financial AI Agents Layer**: Bao gồm Financial Chain-of-Thought (CoT) prompting
2. **Financial LLMs Algorithms Layer**: Cấu hình và sử dụng các mô hình được điều chỉnh đặc biệt
3. **LLMOps and DataOps Layers**: Chiến lược tích hợp đa nguồn
4. **Multi-source LLM Foundation Models Layer**: Hỗ trợ chức năng plug-and-play

### Workflow của AI Agent:

1. **Perception**: Thu thập và diễn giải dữ liệu tài chính đa phương thức
2. **Brain**: Xử lý dữ liệu với LLMs và sử dụng Financial CoT
3. **Action**: Thực hiện các hướng dẫn và áp dụng công cụ

## 📁 Cấu trúc thư mục

```
FinRobot/
├── finrobot/                 # Main package
│   ├── agents/              # AI agents
│   ├── data_source/         # Data sources
│   └── functional/          # Functional modules
├── tutorials_beginner/       # Tutorials cho người mới
├── tutorials_advanced/       # Tutorials nâng cao
├── configs/                 # Configuration files
├── experiments/             # Experimental files
├── requirements.txt         # Dependencies
├── OAI_CONFIG_LIST          # OpenAI configuration
├── config_api_keys          # API keys configuration
└── README_FINROBOT.md       # This file
```

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **Lỗi numpy compatibility**:
   ```bash
   pip install numpy==1.26.4
   ```

2. **Lỗi onnxruntime**:
   ```bash
   pip install onnxruntime
   ```

3. **Lỗi autogen import**:
   ```bash
   pip install pyautogen==0.2.19
   ```

## 📊 API Keys cần thiết

### 1. OpenAI API
- Đăng ký tại: https://platform.openai.com/
- Cần API key để sử dụng các mô hình GPT

### 2. Finnhub API
- Đăng ký tại: https://finnhub.io/
- Cung cấp dữ liệu thị trường tài chính

### 3. FMP API
- Đăng ký tại: https://financialmodelingprep.com/
- Cung cấp dữ liệu tài chính công ty

### 4. SEC API
- Đăng ký tại: https://www.sec.gov/edgar/sec-api-documentation
- Truy cập dữ liệu SEC filings

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📄 License

Dự án này được phát hành dưới giấy phép Apache-2.0.

## ⚠️ Disclaimer

Các mã và tài liệu được cung cấp trong repository này được phát hành dưới giấy phép Apache-2.0. Chúng không nên được hiểu là tư vấn tài chính hoặc khuyến nghị cho giao dịch thực tế. Việc thận trọng và tham khảo ý kiến của các chuyên gia tài chính có trình độ trước khi thực hiện bất kỳ hành động giao dịch hoặc đầu tư nào là điều bắt buộc.

## 📞 Liên hệ

- **Repository**: https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM
- **Author**: Trinh Khuong
- **Email**: [Your Email]

---

<div align="center">
<p>Made with ❤️ for the Vietnamese AI community</p>
</div>
