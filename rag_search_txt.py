import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from chromadb import PersistentClient
from rank_bm25 import BM25Okapi # 👈 키워드 검색(BM25) 라이브러리

# ============================================
# 1. 환경 설정
# ============================================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

genai.configure(api_key=api_key)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "txt_collection"

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash"

# ============================================
# 2. 유틸리티 함수
# ============================================
def get_query_embedding(query: str):
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]
    except Exception as e:
        print(f"❌ 임베딩 오류: {e}")
        return None

def simple_tokenize(text):
    """간단한 띄어쓰기 기반 토크나이저 (한국어 형태소 분석기 없이 사용)"""
    return text.lower().split()

# ============================================
# 3. 하이브리드 검색 (Vector + Keyword)
# ============================================
def hybrid_search(query, collection, k=10):
    """
    ChromaDB(벡터)와 BM25(키워드)를 결합하여 최고의 문서를 찾습니다.
    """
    print("\n---  하이브리드 검색 시작 (Vector + BM25) ---")
    
    # 1. DB에 있는 모든 문서 가져오기 (BM25 인덱싱용)
    # (문서 양이 수만 개가 아니면 매번 로드해도 빠릅니다)
    all_docs_data = collection.get() 
    all_docs = all_docs_data['documents']
    all_ids = all_docs_data['ids']
    
    if not all_docs:
        print(" DB가 비어있습니다.")
        return []

    # -------------------------------------------------------
    # A. BM25 (키워드 검색) 점수 계산
    # -------------------------------------------------------
    tokenized_corpus = [simple_tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = simple_tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 점수 정규화 (0~1 사이로 맞춤)
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)
    
    # -------------------------------------------------------
    # B. Vector (의미 검색) 점수 계산
    # -------------------------------------------------------
    query_embedding = get_query_embedding(query)
    if not query_embedding: return []

    # ChromaDB에서 전체 문서와의 거리를 계산하기 어려우므로,
    # 여기서는 검색된 Top-K만 쓰는 게 아니라, 
    # BM25가 찾은 문서들에 대해 벡터 유사도를 검증하거나 결합하는 방식을 씁니다.
    # 하지만 구현의 편의를 위해 Chroma에서 넓게(Top-100) 가져와서 섞겠습니다.
    
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(all_docs), # 전체 문서 대상 (소규모라 가능)
    )
    
    # 벡터 거리(Distance)를 유사도(Score)로 변환 (거리가 0이면 점수 1)
    # Chroma Distance는 보통 L2(유클리드)나 Cosine 거리
    distances = vector_results['distances'][0]
    vector_ids = vector_results['ids'][0]
    
    # ID:Score 딕셔너리 생성
    vector_score_map = {}
    max_dist = max(distances) if distances else 1
    for doc_id, dist in zip(vector_ids, distances):
        # 거리가 가까울수록(작을수록) 점수가 높아야 함
        score = 1 - (dist / (max_dist + 0.0001)) 
        vector_score_map[doc_id] = score

    # -------------------------------------------------------
    # C. 점수 결합 (Weighted Sum)
    # -------------------------------------------------------
    final_scores = []
    
    # 가중치 설정 (키워드 중요하면 alpha를 높임)
    # 여기서는 키워드(0.6) + 의미(0.4) 정도로 설정
    alpha = 0.6 
    
    for i, doc_id in enumerate(all_ids):
        v_score = vector_score_map.get(doc_id, 0)
        k_score = bm25_scores[i]
        
        # 최종 점수 = (키워드점수 * 0.6) + (벡터점수 * 0.4)
        total_score = (k_score * alpha) + (v_score * (1 - alpha))
        
        # [강의명] 태그가 있으면 가산점 (메타데이터 인젝션 효과 극대화)
        if query.split()[0] in all_docs[i]: # 단순 체크
             total_score += 0.1

        final_scores.append((total_score, all_docs[i]))

    # 점수순 정렬
    final_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 상위 K개 반환
    return [doc for score, doc in final_scores[:k]]

# ============================================
# 4. 메인 실행 로직
# ============================================
def run_rag(query):
    if not os.path.exists(CHROMA_DB_PATH):
        return "DB가 없습니다. 먼저 /build API를 실행하세요."

    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # 1. 하이브리드 검색 실행 (Top 7)
    top_docs = hybrid_search(query, collection, k=7)
    
    if not top_docs:
        return "관련 문서를 찾지 못했습니다."

    # --- Gemini 답변 생성 ---
    context = "\n\n".join(top_docs)
    
    prompt = f"""
    당신은 수강신청 도우미입니다. 
    [관련 문서]를 보고 사용자의 질문에 정확하게 답변하세요.

    [관련 문서]
    {context}

    [질문]
    {query}

    [답변 지침]
    1. 교수님 성함, 과목명 등 고유명사는 문서에 있는 그대로 정확히 말하세요.
    2. 문서에 없는 내용은 "정보가 없습니다"라고 하세요.
    3. 출처가 되는 강의명을 함께 언급해주세요.
    """

    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        resp = model.generate_content(prompt)
        return resp.text   
    except Exception as e:
        return f"Gemini API 오류: {e}"
