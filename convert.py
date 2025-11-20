import os
import google.generativeai as genai
import pymupdf  # PyMuPDF (pip install pymupdf) 설치 필요
from dotenv import load_dotenv
import time

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 설정
PDF_DIR = "pdfs"       # 원본 PDF가 있는 폴더
TXT_DIR = "texts"      # 정제된 TXT를 저장할 폴더

def extract_text_from_pdf(pdf_path):
    """PyMuPDF를 사용하여 PDF에서 텍스트만 빠르게 추출"""
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text_with_gemini(raw_text, filename):
    """Gemini에게 더러운 텍스트를 깔끔한 Markdown으로 정리시킴"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""
    너는 데이터 정리 전문가야. 아래는 PDF에서 추출한 대학교 강의계획서의 원본 텍스트야.
    내용이 뒤죽박죽이고 쓸모없는 정보(페이지 번호, '강의평 없음', 반복되는 헤더 등)가 많아.
    
    이 내용을 **RAG 검색에 최적화된 깔끔한 Markdown 형식**으로 요약 및 정리해줘.
    
    [요구사항]
    1. 파일명 '{filename}'을 고려해서 문서 제목을 적어줘.
    2. 강의명, 교수님, 학년, 평가방법, 주차별 계획 등을 명확한 소제목으로 구분해.
    3. 표로 된 주차별 계획은 리스트 형태로 1주차, 2주차... 로 정리해.
    4. 강의평 보존: - 텍스트 뒷부분에 포함된 **학생들의 '강의평', '수강 후기', '꿀팁' 등을 삭제하지 마.**
       - '## 강의평'이라는 섹션을 따로 만들어서 내용을 정리해줘.
       - 만약 "아직 강의평이 없습니다"라는 문구만 있다면, 해당 섹션에 "등록된 강의평 없음"이라고 적어줘.
    5. 노이즈 제거: 페이지 번호, 반복되는 헤더/푸터, 의미 없는 특수문자는 삭제해.


    [원본 텍스트]
    {raw_text[:30000]} # 토큰 제한 고려하여 앞부분 위주로 (필요시 조절)
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f" Gemini 변환 실패 ({filename}): {e}")
        return raw_text # 실패하면 원본이라도 반환

def process_all_pdfs():
    if not os.path.exists(TXT_DIR):
        os.makedirs(TXT_DIR)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    print(f"총 {len(pdf_files)}개의 PDF를 발견했습니다.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        txt_filename = pdf_file.replace(".pdf", ".txt")
        txt_path = os.path.join(TXT_DIR, txt_filename)
        
        # 이미 변환된 파일은 패스 (이어하기 기능)
        if os.path.exists(txt_path):
            print(f"⏩ 이미 존재함: {txt_filename}")
            continue

        print(f" 처리 중: {pdf_file}...", end="")
        
        # 1. PDF -> Raw Text
        try:
            raw_text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"  PDF 읽기 실패: {e}")
            continue
            
        # 2. Raw Text -> Clean Text (LLM)
        clean_text = clean_text_with_gemini(raw_text, pdf_file)
        
        # 3. 저장
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        
        print("  변환 완료!")
        # API 에러 방지를 위한 짧은 대기
        time.sleep(2)

if __name__ == "__main__":
    # pip install pymupdf google-generativeai python-dotenv
    process_all_pdfs()