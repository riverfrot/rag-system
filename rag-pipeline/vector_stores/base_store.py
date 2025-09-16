#!/usr/bin/env python3
"""
Base Vector Store Interface
ChromaDB와 Pinecone을 통합한 추상 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVectorStore(ABC):
    """벡터스토어 기본 인터페이스"""
    
    @abstractmethod
    def initialize(self) -> None:
        """벡터스토어 초기화"""
        pass
    
    @abstractmethod
    def upsert_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """문서를 벡터스토어에 업서트"""
        pass
    
    @abstractmethod
    def similarity_search(self, 
                         query: str, 
                         k: int = 5, 
                         filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """유사도 검색"""
        pass
    
    @abstractmethod
    def delete_all(self) -> None:
        """모든 벡터 삭제"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """벡터스토어 통계"""
        pass