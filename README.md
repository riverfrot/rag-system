# RAG System for Issue Resolution

이 프로젝트는 특정 레포지토리에서 이슈를 자동으로 해결하는 RAG (Retrieval-Augmented Generation) 시스템입니다.
현재는 https://github.com/riverfrot/sample-spring 기준이나 추후 변경 예정 입니다.

## 시스템 구조

```
Repository → Ingestion → Chunking → Embedding → VectorStore
                                                      ↓
User Query → Vector Search → Context Retrieval → LLM → Response
                ↓
         External Research (Tavily API)
```

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일에 필요한 API 키들이 설정되어 있는지 확인:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

## 사용 방법

### 1. Repository Ingestion (저장소 임베딩)

```bash
# sample-spring 레포지토리 임베딩
python main.py ingest --repo https://github.com/riverfrot/sample-spring

# 커스텀 저장 경로 사용
python main.py ingest --repo https://github.com/riverfrot/sample-spring --persist-dir ./my_db
```

### 2. Issue Query (이슈 해결)

```bash
# 이슈 해결 쿼리
python main.py query --issue "ISSUE-2: 데이터 영속성 문제"

# 상세한 이슈 설명
python main.py query --issue "Spring Boot 애플리케이션에서 데이터베이스 연결 문제 해결"

# 커스텀 데이터베이스 경로 사용
python main.py query --issue "데이터베이스 설정 문제" --persist-dir ./my_db
```

### 3. 개별 모듈 실행

```bash
# 직접 ingestion 실행
python rag-pipeline/ingest.py --repo https://github.com/riverfrot/sample-spring

# 직접 query 실행  
python rag-pipeline/query.py --issue "Spring Boot database issue"
```

## 주요 기능

### 🔄 Repository Ingestion
- GitHub 레포지토리 자동 클론
- 코드 파일 자동 감지 (.kt, .java, .md, .yml 등)
- 스마트 청킹으로 코드 분할
- OpenAI text-embedding-3-large로 임베딩
- ChromaDB에 벡터 저장

### 🔍 Intelligent Query
- 벡터 유사도 검색으로 관련 코드 추출
- Tavily API를 통한 외부 리서치
- GPT-4를 활용한 종합적 해결책 생성
- 한국어 응답 지원

### 📁 지원 파일 형식
- Kotlin: `.kt`
- Java: `.java`
- Documentation: `.md`, `.txt`
- Configuration: `.yml`, `.yaml`, `.json`, `.xml`, `.properties`
- Build files: `.gradle`

## 프로젝트 구조

```
rag-system/
├── main.py                 # 통합 CLI 인터페이스
├── rag-pipeline/
│   ├── __init__.py
│   ├── ingest.py          # 레포지토리 임베딩 파이프라인
│   └── query.py           # 이슈 해결 쿼리 시스템
├── requirements.txt       # Python 의존성
├── .env                  # 환경 변수 (API 키)
└── README.md             # 프로젝트 문서
```

## 예시 워크플로우

### 1. 레포지토리 임베딩
```bash
$ python main.py ingest --repo https://github.com/riverfrot/sample-spring

🚀 Starting Repository Ingestion...
📥 Cloning repository: https://github.com/riverfrot/sample-spring
✅ Repository cloned to: ./temp_repo
🔍 Extracting code files from: ./temp_repo
✅ Found 15 code files
📄 Processing 15 files into chunks...
✅ Created 45 document chunks
🔄 Creating vector store with 45 chunks...
✅ Vector store created and saved to: ./chroma_db
🧹 Cleaned up temporary files
🎉 Repository ingestion completed successfully!
```

### 2. 이슈 해결
```bash
$ python main.py query --issue "ISSUE-2: 데이터 영속성 문제"

🎯 Starting Issue Resolution...
📚 Loading vector store from: ./chroma_db
✅ Vector store loaded successfully
🔍 Searching for relevant context: 'ISSUE-2: 데이터 영속성 문제'
✅ Found 5 relevant code snippets
🌐 Conducting external research: 'spring boot kotlin ISSUE-2: 데이터 영속성 문제 solution'
✅ External research completed (3 results)
🤖 Generating solution using OpenAI GPT-4...
✅ Solution generated successfully
🎉 Issue resolution completed!

================================================================================
📋 ISSUE RESOLUTION REPORT
================================================================================

🎯 Issue: ISSUE-2: 데이터 영속성 문제

📁 Relevant Files (5):
  • ./temp_repo/src/main/kotlin/com/example/IssueService.kt
  • ./temp_repo/src/main/resources/application.yml
  • ./temp_repo/README.md
  • ./temp_repo/build.gradle
  • ./temp_repo/src/main/kotlin/com/example/Application.kt

🤖 Solution:
----------------------------------------
1. 문제 분석
현재 시스템에서는 in-memory List를 사용하여 데이터를 저장하고 있어, 
애플리케이션 재시작 시 데이터가 손실되는 영속성 문제가 발생하고 있습니다.

2. 원인 파악
IssueService.kt에서 데이터를 메모리상의 리스트에만 저장하고 있어 
애플리케이션 종료 시 모든 데이터가 사라집니다.

3. 해결 방안
Spring Data JPA와 H2 데이터베이스를 도입하여 데이터 영속성을 구현합니다.

[상세한 코드 예시와 구현 방법 포함...]

================================================================================
```

## 고급 설정

### 커스텀 임베딩 모델 사용

코드에서 직접 HuggingFace 모델로 변경 가능:

```python
from langchain.embeddings import HuggingFaceEmbeddings

# 코드 전용 임베딩 모델
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/codebert-base")

# 또는 범용 모델
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
```

### 벡터 데이터베이스 위치 변경

```bash
# 환경 변수로 설정
export CHROMA_PERSIST_DIRECTORY="/path/to/custom/db"

# 또는 CLI 옵션 사용
python main.py ingest --repo https://github.com/example/repo --persist-dir /custom/path
```

## 문제 해결

### 일반적인 오류들

1. **API 키 관련 오류**
   ```
   ValueError: OPENAI_API_KEY not found in environment variables
   ```
   → `.env` 파일에 올바른 API 키가 설정되어 있는지 확인

2. **Vector Store 로드 오류**
   ```
   Error loading vector store
   ```
   → 먼저 `ingest` 명령어로 레포지토리를 임베딩했는지 확인

3. **레포지토리 클론 실패**
   ```
   Error cloning repository
   ```
   → 레포지토리 URL이 올바르고 접근 가능한지 확인

### 디버깅

상세한 로그를 보려면 Python 로깅을 활성화:

```bash
export PYTHONPATH=. 
python -v main.py ingest --repo https://github.com/example/repo
```

## Roadmap

**TODO:**
- [ ] Vector Database 성능 최적화 (Pinecone/Chroma)
- [ ] Memory 관리 및 conversation chain 고도화
- [ ] Error handling 및 retry 로직 추가
- [ ] API 응답 시간 모니터링 추가

**Detail Message:**
```python
# 추가해야 할 기능들
1. Multi-modal AI 연동 (이미지, 문서 처리)
2. Custom Tool 개발 (웹 크롤링, 데이터 분석)
3. 실시간 성능 대시보드
4. 비용 최적화 로직
```

## 라이선스

MIT License
