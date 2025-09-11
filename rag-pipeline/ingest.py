#!/usr/bin/env python3
"""
Repository Ingestion Pipeline
Repo 클론 후 코드 파일 수집 → 청킹 & 임베딩 → VectorStore 저장
"""

import os
import sys
import git
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import chromadb

load_dotenv()

# LangSmith 추적 활성화
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-system")

class RepositoryIngestor:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # OpenAI text-embedding-3-large 사용
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=self.openai_api_key
        )
        
        # 텍스트 스플리터 설정 (코드에 최적화)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    def clone_repository(self, repo_url: str, local_path: str = "./temp_repo") -> str:
        """GitHub 레포지토리 클론"""
        print(f"📥 Cloning repository: {repo_url}")
        
        # 기존 디렉토리가 있다면 삭제
        if os.path.exists(local_path):
            import shutil
            shutil.rmtree(local_path)
        
        try:
            git.Repo.clone_from(repo_url, local_path)
            print(f"✅ Repository cloned to: {local_path}")
            return local_path
        except Exception as e:
            print(f"❌ Error cloning repository: {e}")
            raise
    
    def extract_code_files(self, repo_path: str) -> List[str]:
        """코드 파일 추출 (.kt, .java, .md 등)"""
        print(f"🔍 Extracting code files from: {repo_path}")
        
        # 지원하는 파일 확장자
        code_extensions = {
            '.kt', '.java', '.py', '.js', '.ts', '.md', '.txt', 
            '.yml', '.yaml', '.json', '.xml', '.gradle', '.properties'
        }
        
        files = []
        ignored_dirs = {'.git', 'node_modules', '__pycache__', '.gradle', 'build', 'target'}
        
        for root, dirs, filenames in os.walk(repo_path):
            # 무시할 디렉토리 제외
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                file_ext = Path(filename).suffix.lower()
                
                if file_ext in code_extensions:
                    files.append(file_path)
        
        print(f"✅ Found {len(files)} code files")
        return files
    
    def process_files_to_chunks(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """파일을 청크로 분할하고 메타데이터 추가"""
        print(f"📄 Processing {len(file_paths)} files into chunks...")
        
        documents = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # 파일 내용을 청크로 분할
                chunks = self.text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'metadata': {
                            'source': file_path,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'file_type': Path(file_path).suffix,
                            'file_name': Path(file_path).name,
                            'relative_path': str(Path(file_path).relative_to(Path(file_path).parts[0]))
                        }
                    })
            
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                continue
        
        print(f"✅ Created {len(documents)} document chunks")
        return documents
    
    def create_vector_store(self, documents: List[Dict[str, Any]]) -> None:
        """ChromaDB 벡터스토어 생성 및 저장"""
        print(f"🔄 Creating vector store with {len(documents)} chunks...")
        
        if not documents:
            print("❌ No documents to process")
            return
        
        # 텍스트와 메타데이터 분리
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        try:
            # ChromaDB 벡터스토어 생성
            vectorstore = Chroma.from_texts(
                texts=texts,
                metadatas=metadatas,
                embedding=self.embeddings,
                persist_directory=self.chroma_persist_dir
            )
            
            # 벡터스토어 저장
            vectorstore.persist()
            print(f"✅ Vector store created and saved to: {self.chroma_persist_dir}")
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    
    def ingest_repository(self, repo_url: str) -> None:
        """전체 ingestion 파이프라인 실행"""
        print("🚀 Starting repository ingestion pipeline...")
        
        try:
            # 1. 레포지토리 클론
            repo_path = self.clone_repository(repo_url)
            
            # 2. 코드 파일 추출
            file_paths = self.extract_code_files(repo_path)
            
            # 3. 파일을 청크로 분할
            documents = self.process_files_to_chunks(file_paths)
            
            # 4. 벡터스토어 생성
            self.create_vector_store(documents)
            
            # 5. 임시 파일 정리
            import shutil
            shutil.rmtree(repo_path)
            print("🧹 Cleaned up temporary files")
            
            print("🎉 Repository ingestion completed successfully!")
            
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Repository Ingestion Pipeline")
    parser.add_argument("--repo", required=True, help="Repository URL to ingest")
    parser.add_argument("--persist-dir", help="ChromaDB persist directory", 
                       default="./chroma_db")
    
    args = parser.parse_args()
    
    # 환경 변수 설정
    if args.persist_dir:
        os.environ["CHROMA_PERSIST_DIRECTORY"] = args.persist_dir
    
    # Ingestion 실행
    ingestor = RepositoryIngestor()
    ingestor.ingest_repository(args.repo)

if __name__ == "__main__":
    main()