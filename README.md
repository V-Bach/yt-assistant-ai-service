# Youtube Learning Assistant - AI Analysis Service (Python Backend)

## Tổng quan học thuật (Academic Overview)

Dự án này đóng vai trò là **AI Engine** trong hệ thống hỗ trợ học tập qua video. Mục tiêu cốt lõi là chuyển đổi dữ liệu thô từ video (Transcripts) thành các cấu trúc kiến thức có hệ thống thông qua các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) tiên tiến.

Repo này tập trung vào việc triển khai quy trình **RAG (Retrieval-Augmented Generation)**, giúp giải quyết vấn đề "ảo tưởng" (hallucination) của AI bằng cách kết hợp sức mạnh lập luận của LLM với cơ sở dữ liệu tri thức được cá nhân hóa từ người dùng.

## Kiến trúc kết nối (System Integration)

Hệ thống hoạt động theo mô hình **Distributed Microservices Architecture** (Kiến trúc vi dịch vụ phân tán), trong đó Repo này đóng vai trò trung tâm xử lý dữ liệu:

1. **Kết nối với Chrome Extension (Frontend):**
* Python Backend cung cấp các RESTful API (Cổng 8000) để Extension gửi dữ liệu Transcript.
* Nó thực hiện băm nhỏ dữ liệu (Chunking), nhúng (Embedding) và trả về kết quả tóm tắt Markdown.


2. **Kết nối với C# Backend (Data Manager):**
* Sau khi Python hoàn tất phân tích, kết quả được Chrome Extension làm trung gian đẩy sang C# (Cổng 5104) để lưu trữ vĩnh viễn vào MySQL.
* Điều này tách biệt hoàn toàn giữa **Tính toán (AI - Python)** và **Lưu trữ (Persistence - C#)**, giúp hệ thống có khả năng mở rộng (Scalability) cực cao.

## Công nghệ sử dụng (Tech Stack)

* **Framework:** FastAPI (Hiệu năng cao, hỗ trợ Async).
* **LLM:** Google Gemini API (Gemini 1.5 Flash).
* **Orchestration:** LangChain (Quản lý luồng AI và Prompt Template).
* **Vector Database:** ChromaDB (Lưu trữ và truy xuất vector kiến thức).
* **Embeddings:** Google Text Embedding-004.

## Hướng dẫn cài đặt (Setup Guide)

### 1. Chuẩn bị môi trường

Yêu cầu Python bản 3.10 trở lên.

### 2. Cài đặt các thư viện cần thiết

Mở terminal tại thư mục gốc của dự án và chạy các lệnh sau:

```bash
# Tạo môi trường ảo (Khuyến khích)
python -m venv venv
source venv/bin/activate  # Trên Linux/Mac
# venv\Scripts\activate   # Trên Windows

# Cài đặt các thư viện
pip install fastapi uvicorn langchain-google-genai langchain-chroma langchain-text-splitters pydantic-settings python-dotenv mysql-connector-python

```

### 3. Cấu hình biến môi trường (Environment Variables)

Tạo một file tên là `.env` trong thư mục gốc và dán nội dung sau vào:

```env
GOOGLE_API_KEY= (Điền key của bạn ở đây/Lấy key qua Google AI Studio)

```


### 4. Khởi chạy Server

Chạy lệnh uvicorn để server bắt đầu lắng nghe yêu cầu:

```bash
uvicorn main:app --reload --port 8000

```

* Server sẽ chạy tại địa chỉ: `http://localhost:8000`.

---

## Các Endpoint chính

* `POST /ai/process`: Nhận transcript video, lưu vào ChromaDB và trả về bản tóm tắt AI.
* `POST /ai/ask-anything`: Nhận câu hỏi từ người dùng và sử dụng RAG để trả lời dựa trên kiến thức video đã xem.
* `GET /`: Kiểm tra trạng thái hoạt động của server.

