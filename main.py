import os
import time
import traceback
import mysql.connector
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from fastapi.middleware.cors import CORSMiddleware



load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_db = Chroma(
    persist_directory="./db_knowledge",
    embedding_function=embeddings
)


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest", 
    temperature=0.3,
    max_retries=2, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",          
        password="vuthebach",  
        database="yt_learning_db"   
    )


template = """
Bạn là một Chuyên gia Giáo dục và Nhà khoa học dữ liệu. 
Nhiệm vụ: Phân tích nội dung video dựa trên Transcript được cung cấp nhưng KHÔNG BỊ LỆ THUỘC hoàn toàn vào nó.

QUY TRÌNH XỬ LÝ:
1. Xác định các Keyword (Từ khóa) và Chủ đề chính mà video "{title}" đang đề cập.
2. Sử dụng kiến thức chuyên sâu của bạn (World Knowledge) để giải thích chi tiết và đầy đủ nhất về các khái niệm đó. 
3. Nếu Transcript bị thiếu hụt hoặc sai lỗi chính tả, hãy dựa vào ngữ cảnh và tiêu đề để khôi phục kiến thức chuẩn.
4. Đảm bảo bản phân tích có cấu trúc sư phạm, dễ hiểu và mang tính chuyên gia.

CẤU TRÚC ĐẦU RA (Markdown):
# 🎯 BẢN CHẤT KIẾN THỨC: {title}
> (Giải thích giá trị thực tế của kiến thức này)

## 🧠 PHÂN TÍCH CHI TIẾT (Theo khung sườn video)
### 1. [Chủ đề 1]
- **Giải mã:** (Giải thích chi tiết khái niệm, công thức, hoặc logic đằng sau)
- **Kiến thức mở rộng:** (Những kiến thức chuyên sâu ngoài transcript)
- **Ví dụ minh họa:** (Ví dụ thực tế)

## 💡 TỔNG KẾT & LỜI KHUYÊN HÀNH ĐỘNG
- (Quy trình áp dụng kiến thức này vào thực tế)

Transcript tham khảo:
{transcript}
"""

class VideoData(BaseModel):
    videoId: str
    title: str
    transcript: str

@app.get("/")
async def root():
    return {"message": "Python AI Service is Running!"}

import time
from tenacity import retry, stop_after_attempt, wait_exponential


@app.post("/ai/process")
async def process_video(data: VideoData):
    print(f"\n🚀 [BẮT ĐẦU] --- Phân tích: {data.title} ---")
    try:
        print("📍 Bước 1: Đang băm transcript và nạp vào ChromaDB...")
        try:
            text_splitter_db = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs_db = text_splitter_db.create_documents(
                [data.transcript], 
                metadatas=[{"title": data.title, "videoId": data.videoId}] 
            )
            vector_db.add_documents(docs_db)
            print("✅ Bước 1 thành công: Đã lưu kiến thức vào Vector DB.")
        except Exception as e_db:
            print(f"❌ LỖI BƯỚC 1 (Vector DB): {str(e_db)}")
            return {"status": "error", "message": f"Lỗi lưu Database kiến thức: {str(e_db)}"}

        print("📍 Bước 2: Đang đóng gói Prompt chuyên gia...")
        full_context = data.transcript[:15000] 
        prompt_info = PromptTemplate(template=template, input_variables=["title", "transcript"])
        final_prompt = prompt_info.format(title=data.title, transcript=full_context)
        print("✅ Bước 2 thành công: Prompt đã sẵn sàng.")

        print(f"📍 Bước 3: Đang gọi Gemini ({llm.model}) - Vui lòng đợi...")
        try:
            start_time = time.time()
            response = llm.invoke(final_prompt)
            duration = time.time() - start_time
            
            if response and response.content:
                print(f"✅ Bước 3 thành công: AI đã phản hồi sau {duration:.2f} giây.")
                return {
                    "status": "success",
                    "ai_analysis": response.content 
                }
            else:
                print("⚠️ CẢNH BÁO: AI kết nối thành công nhưng trả về nội dung RỖNG.")
                return {"status": "error", "message": "AI phản hồi rỗng (Empty Content)."}
                
        except Exception as e_ai:
            print(f"❌ LỖI BƯỚC 3 (Gemini API): {str(e_ai)}")
            if "429" in str(e_ai) or "RESOURCE_EXHAUSTED" in str(e_ai):
                return {"status": "error", "message": "Hết Quota (429). Đợi 30s rồi bấm lại bro nhé!"}
            return {"status": "error", "message": f"Lỗi từ Gemini: {str(e_ai)}"}

    except Exception as e_main:
        import traceback
        print(f"🔥 LỖI HỆ THỐNG TỔNG THỂ:\n{traceback.format_exc()}")
        return {"status": "error", "message": f"Lỗi không xác định: {str(e_main)}"}

class QuestionRequest(BaseModel):
    question: str

@app.post("/ai/ask-anything")
async def ask_anything(data: QuestionRequest):
    print(f"\n🔍 [RAG] --- Nhận câu hỏi: {data.question} ---")
    try:
        related_docs = vector_db.similarity_search(data.question, k=3)
        
        if related_docs:
            context = "\n---\n".join([d.page_content for d in related_docs])
            print(f"✅ Tìm thấy {len(related_docs)} đoạn liên quan.")
        else:
            context = "Không có thông tin cụ thể trong database."
            print("⚠️ Database không có gì liên quan.")

        rag_prompt = f"""
        Bạn là trợ lý học tập. Dựa vào kiến thức sau: {context}
        ---
        Câu hỏi: {data.question}
        Yêu cầu: Trả lời ngắn gọn bằng Markdown. Nếu không có thông tin, hãy dùng kiến thức chuyên gia của bạn.
        """

        print("🤖 Đang đợi Gemini 2.0 phản hồi...")
        response = llm.invoke(rag_prompt)
        
        ai_text = ""
        if response and response.content:
            if isinstance(response.content, list):
                ai_text = " ".join([block['text'] for block in response.content if 'text' in block])
            else:
                ai_text = str(response.content)
            
            print(f"✅ AI trả lời thành công ({len(ai_text)} ký tự).")
            return {"answer": ai_text}
        
        print("⚠️ AI trả về rỗng.")
        return {"answer": "AI không trả về nội dung."}

    except Exception as e:
        error_msg = str(e)
        print(f"❌ LỖI RAG: {error_msg}")
        
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {"answer": "QUOTA_EXCEEDED"} 
            
        return {"answer": f"Lỗi hệ thống: {error_msg}"}
    
@app.get("/api/video/history")
async def get_video_history():
    print("[DEBUG] 📂 Đang lôi lịch sử video từ MySQL...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = "SELECT videoId, title, summary, createdAt FROM videos ORDER BY createdAt DESC"
        cursor.execute(query)
        
        data = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        print(f"✅ Đã tìm thấy {len(data)} video trong lịch sử.")
        return data  

    except Exception as e:
        print(f"❌ Lỗi truy vấn MySQL: {str(e)}")
        return [] 
