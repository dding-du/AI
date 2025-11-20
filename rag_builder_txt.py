import os
import re
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb import PersistentClient

# ===============================================
# 환경 설정
# ===============================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

genai.configure(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = os.path.join(BASE_DIR, "new2_texts")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "txt_collection"

EMBEDDING_MODEL = "models/text-embedding-004"

# ===============================================
# 1. 핵심 정보 추출 함수 (Metadata Parser)
# ===============================================
def extract_core_info(text):
    """
    강의계획서 텍스트에서 담당교수, 학년, 학점 등을 정규식으로 추출합니다.
    """
    info = {
        "professor": "정보없음",
        "grade": "정보없음",
        "credit": "정보없음"
    }
    
    # 1. 담당교수 추출 (예: * **담당교수:** 오민식)
    prof_match = re.search(r"\*\*담당교수:\*\*\s*([^\n]+)", text)
    if prof_match:
        info["professor"] = prof_match.group(1).strip()

    # 2. 대상학년 추출 (예: * **대상학년:** 3학년)
    grade_match = re.search(r"\*\*대상학년:\*\*\s*([^\n]+)", text)
    if grade_match:
        info["grade"] = grade_match.group(1).strip()

    # 3. 학점 추출
    credit_match = re.search(r"\*\*학점/시간:\*\*\s*([^\n]+)", text)
    if credit_match:
        info["credit"] = credit_match.group(1).strip()
        
    return info

# ===============================================
# 2. 파일 로드 및 청크 생성 (정보 주입)
# ===============================================
def load_and_chunk_files(txt_files):
    all_chunks = []
    all_metadatas = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    print(f"--- 파일 처리 및 정보 추출 시작 ({len(txt_files)}개) ---")

    for path in txt_files:
        try:
            file_name = os.path.splitext(os.path.basename(path))[0]
            
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            # 🔥 핵심: 파일에서 중요 정보 미리 뽑기
            info = extract_core_info(raw_text)
            
            # 청크 생성
            chunks = splitter.split_text(raw_text)

            # 🔥 핵심: 모든 청크에 정보 주입 (헤더 업그레이드)
            for chunk in chunks:
                # 예: [강의명: 통계적데이터분석 | 교수: 오민식 | 학년: 3학년]
                header_tag = f"[강의명: {file_name} | 교수: {info['professor']} | 학년: {info['grade']}]"
                enhanced_chunk = f"{header_tag}\n{chunk}"
                
                all_chunks.append(enhanced_chunk)
                
                # 메타데이터에도 저장 (나중에 필터링 가능하도록)
                all_metadatas.append({
                    "source": file_name,
                    "professor": info['professor'],
                    "grade": info['grade']
                })

            print(f"✅ {file_name} -> 교수: {info['professor']}")

        except Exception as e:
            print(f"❌ 파일 오류 {path}: {e}")

    return all_chunks, all_metadatas

# ===============================================
# 3. 임베딩 생성
# ===============================================
def get_embeddings_for_chunks(chunks):
    embeddings = []
    total = len(chunks)
    batch_size = 10 

    print(f"\n--- 임베딩 생성 시작 (총 {total}개) ---")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document",
            )
            if 'embedding' in result:
                embeddings.extend(result['embedding'])
            
            print(f"  → {min(i + batch_size, total)}/{total} 처리 완료", end="\r")
            time.sleep(1)  
        except Exception as e:
            print(f"\n❌ 배치 오류: {e}")
            continue

    print("\n✅ 임베딩 생성 완료")
    return embeddings

# ===============================================
# 4. ChromaDB 저장
# ===============================================
def build_rag_database():
    if not os.path.exists(TXT_DIR):
        os.makedirs(TXT_DIR)
        print(f"📁 '{TXT_DIR}' 폴더가 없어서 생성했습니다.")
        return

    txt_files = [
        os.path.join(TXT_DIR, f) for f in os.listdir(TXT_DIR) if f.lower().endswith(".txt")
    ]

    if not txt_files:
        print(f"❌ '{TXT_DIR}' 폴더에 파일이 없습니다.")
        return

    # 1. 로드 및 정보 주입
    chunks, metadatas = load_and_chunk_files(txt_files)
    
    # 2. 임베딩
    embeddings = get_embeddings_for_chunks(chunks)

    if len(embeddings) != len(chunks):
        print("❌ 임베딩 개수 오류")
        return

    # 3. 저장
    print("\n--- ChromaDB 저장 중 ---")
    try:
        client = PersistentClient(path=CHROMA_DB_PATH)
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing:
            client.delete_collection(COLLECTION_NAME)
            print("🗑  기존 DB 삭제됨")

        collection = client.get_or_create_collection(COLLECTION_NAME)
        
        ids = [f"doc_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"✅ 저장 완료 (총 {collection.count()}개)")

    except Exception as e:
        print(f"❌ DB 저장 오류: {e}")

if __name__ == "__main__":
    build_rag_database()