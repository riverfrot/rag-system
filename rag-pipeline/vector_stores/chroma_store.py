#!/usr/bin/env python3
"""
ChromaDB Vector Store Implementation
ChromaDB를 사용한 로컬 벡터 검색 시스템
"""

import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
import chromadb

from .base_store import BaseVectorStore

load_dotenv()


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB 벡터스토어 클래스"""
    
    def __init__(self, 
                 collection_name: str = "rag-system",
                 persist_directory: str = "./chroma_db"):
        """
        ChromaDB 벡터스토어 초기화
        
        Args:
            collection_name: 컬렉션 이름
            persist_directory: 데이터 저장 디렉토리
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # API 키 확인
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # OpenAI Embeddings 초기화
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.openai_api_key
        )
        
        self.vectorstore = None
        self.client = None
    
    def initialize(self) -> None:
        """전체 초기화 프로세스"""
        print("🚀 Initializing ChromaDB Vector Store...")
        
        try:
            # ChromaDB 클라이언트 초기화
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # LangChain ChromaDB 초기화
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print(f"📁 Using persist directory: {self.persist_directory}")
            print(f"📋 Collection: {self.collection_name}")
            
            # 기존 데이터 확인
            stats = self.get_stats()
            print(f"📊 Existing vectors: {stats.get('total_vectors', 0)}")
            
            print("✅ ChromaDB Vector Store initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
            raise
    
    def upsert_documents(self, 
                        documents: List[Dict[str, Any]], 
                        batch_size: int = 100) -> None:
        """
        문서를 벡터스토어에 업서트
        
        Args:
            documents: 문서 리스트 (content, metadata 포함)
            batch_size: 배치 크기
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")
            
        print(f"📤 Upserting {len(documents)} documents to ChromaDB...")
        
        try:
            # 텍스트와 메타데이터 분리
            texts = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            # 고유 ID 생성
            ids = [f"doc_{i}_{int(time.time())}" for i in range(len(documents))]
            
            # 배치 처리
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                self.vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                print(f"✅ Upserted batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            # 벡터스토어 영구 저장
            self.vectorstore.persist()
            print(f"🎉 Successfully upserted {len(documents)} documents")
            
        except Exception as e:
            print(f"❌ Error upserting documents: {e}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5, 
                         filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        유사도 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            filter_dict: 메타데이터 필터
            
        Returns:
            검색 결과 리스트
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")
            
        print(f"🔍 Searching for: '{query}' (top {k})\"")
        
        try:
            # ChromaDB 검색 (점수 포함)
            results_with_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            # 결과 변환
            results = []
            for doc, score in results_with_scores:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score)
                })
            
            print(f"✅ Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            raise
    
    def delete_all(self) -> None:
        """모든 벡터 삭제"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")
            
        print("🗑️ Deleting all vectors from ChromaDB...")
        try:
            # 컬렉션 삭제 후 재생성
            if self.client:
                try:
                    self.client.delete_collection(self.collection_name)
                    print("✅ Collection deleted")
                except Exception:
                    print("⚠️ Collection not found or already empty")
            
            # 벡터스토어 재초기화
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print("✅ All vectors deleted and collection recreated")
            
        except Exception as e:
            print(f"❌ Error deleting vectors: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """벡터스토어 통계 반환"""
        try:
            if self.client:
                collection = self.client.get_collection(self.collection_name)
                count = collection.count()
                return {
                    'total_vectors': count,
                    'collection_name': self.collection_name,
                    'persist_directory': self.persist_directory
                }
            return {'total_vectors': 0}
            
        except Exception as e:
            print(f"⚠️ Error getting stats: {e}")
            return {'total_vectors': 0}
    
    def get_collection_info(self) -> Dict:
        """컬렉션 상세 정보"""
        if not self.client:
            return {}
            
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                'name': collection.name,
                'count': collection.count(),
                'metadata': collection.metadata
            }
        except Exception as e:
            print(f"⚠️ Error getting collection info: {e}")
            return {}


def main():
    """테스트용 메인 함수"""
    try:
        # ChromaDB 벡터스토어 초기화
        chroma_store = ChromaVectorStore(
            collection_name="test-collection",
            persist_directory="./test_chroma_db"
        )
        
        # 초기화
        chroma_store.initialize()
        
        # 테스트 문서
        test_docs = [
            {
                'content': 'This is a test document about machine learning.',
                'metadata': {'source': 'test.py', 'file_type': '.py'}
            },
            {
                'content': 'Python is a programming language used for AI development.',
                'metadata': {'source': 'python.md', 'file_type': '.md'}
            }
        ]
        
        # 문서 업서트
        chroma_store.upsert_documents(test_docs)
        
        # 검색 테스트
        results = chroma_store.similarity_search("machine learning", k=2)
        
        print("\n🔍 Search Results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Content: {result['content'][:100]}...")
            print(f"   Metadata: {result['metadata']}")
        
        # 통계 출력
        stats = chroma_store.get_stats()
        print(f"\n📊 Stats: {stats}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()