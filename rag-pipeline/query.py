#!/usr/bin/env python3
"""
Issue Query & Resolution Pipeline
Issue 번호 입력 → VectorStore 검색 → Tavily API 리서치 → OpenAI ChatCompletion으로 해결책 생성
"""

import os
import sys
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from tavily import TavilyClient

load_dotenv()

class IssueQuerySystem:
    def __init__(self):
        # API 키 확인
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        # OpenAI 클라이언트 초기화
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Embeddings 초기화 (ingest.py와 동일한 모델 사용)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=self.openai_api_key
        )
        
        # Tavily 클라이언트 초기화
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        
        # ChromaDB 설정
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        self.vectorstore = None
    
    def load_vector_store(self) -> None:
        """저장된 벡터스토어 로드"""
        print(f"📚 Loading vector store from: {self.chroma_persist_dir}")
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.chroma_persist_dir,
                embedding_function=self.embeddings
            )
            print("✅ Vector store loaded successfully")
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            print("💡 Make sure you've run ingest.py first to create the vector store")
            raise
    
    def search_relevant_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """VectorStore에서 관련된 코드 스니펫 검색"""
        print(f"🔍 Searching for relevant context: '{query}'")
        
        if not self.vectorstore:
            self.load_vector_store()
        
        try:
            # 유사도 검색 수행
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            context_docs = []
            for doc, score in results:
                context_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                })
            
            print(f"✅ Found {len(context_docs)} relevant code snippets")
            return context_docs
            
        except Exception as e:
            print(f"❌ Error during vector search: {e}")
            return []
    
    def tavily_research(self, query: str) -> str:
        """Tavily API를 사용한 외부 리서치"""
        print(f"🌐 Conducting external research: '{query}'")
        
        try:
            # Tavily 검색 실행
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_answer=True
            )
            
            research_content = ""
            
            # 답변이 있는 경우
            if response.get('answer'):
                research_content += f"Research Summary: {response['answer']}\n\n"
            
            # 검색 결과 처리
            for i, result in enumerate(response.get('results', []), 1):
                research_content += f"Result {i}:\n"
                research_content += f"Title: {result.get('title', 'N/A')}\n"
                research_content += f"Content: {result.get('content', 'N/A')}\n"
                research_content += f"URL: {result.get('url', 'N/A')}\n\n"
            
            print(f"✅ External research completed ({len(response.get('results', []))} results)")
            return research_content
            
        except Exception as e:
            print(f"⚠️ External research failed: {e}")
            return "External research unavailable due to API error."
    
    def generate_solution(self, issue_description: str, context_docs: List[Dict], research_content: str) -> str:
        """OpenAI ChatCompletion으로 해결책 생성"""
        print("🤖 Generating solution using OpenAI GPT-4...")
        
        # 코드 컨텍스트 준비
        code_context = ""
        for i, doc in enumerate(context_docs, 1):
            code_context += f"Code Snippet {i}:\n"
            code_context += f"File: {doc['metadata'].get('source', 'Unknown')}\n"
            code_context += f"File Type: {doc['metadata'].get('file_type', 'Unknown')}\n"
            code_context += f"Similarity Score: {doc['similarity_score']:.3f}\n"
            code_context += f"Content:\n{doc['content']}\n"
            code_context += "-" * 50 + "\n"
        
        # 프롬프트 구성
        system_prompt = """당신은 Spring Boot와 Kotlin에 전문성을 가진 시니어 개발자입니다. 
주어진 이슈에 대해 코드 분석을 바탕으로 정확하고 실용적인 해결책을 제시해주세요.

응답 형식:
1. 문제 분석
2. 원인 파악  
3. 해결 방안
4. 코드 예시 (필요한 경우)
5. 추가 권장사항

한국어로 답변해주세요."""

        user_prompt = f"""
이슈 설명:
{issue_description}

관련 코드 컨텍스트:
{code_context}

외부 리서치 결과:
{research_content}

위 정보를 바탕으로 이슈에 대한 종합적인 해결책을 제시해주세요.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            solution = response.choices[0].message.content
            print("✅ Solution generated successfully")
            return solution
            
        except Exception as e:
            print(f"❌ Error generating solution: {e}")
            return f"해결책 생성 중 오류가 발생했습니다: {e}"
    
    def resolve_issue(self, issue_description: str) -> Dict[str, Any]:
        """전체 이슈 해결 파이프라인"""
        print(f"🎯 Starting issue resolution for: {issue_description}")
        
        # 1. 관련 코드 컨텍스트 검색
        context_docs = self.search_relevant_context(issue_description)
        
        # 2. 외부 리서치 수행
        research_query = f"spring boot kotlin {issue_description} solution"
        research_content = self.tavily_research(research_query)
        
        # 3. 해결책 생성
        solution = self.generate_solution(issue_description, context_docs, research_content)
        
        # 결과 정리
        result = {
            'issue': issue_description,
            'relevant_files': [doc['metadata'].get('source', 'Unknown') for doc in context_docs],
            'context_docs': context_docs,
            'research_content': research_content,
            'solution': solution
        }
        
        print("🎉 Issue resolution completed!")
        return result
    
    def print_resolution_report(self, result: Dict[str, Any]) -> None:
        """해결책 리포트 출력"""
        print("\n" + "="*80)
        print("📋 ISSUE RESOLUTION REPORT")
        print("="*80)
        
        print(f"\n🎯 Issue: {result['issue']}")
        
        print(f"\n📁 Relevant Files ({len(result['relevant_files'])}):")
        for file in result['relevant_files'][:5]:  # 최대 5개만 표시
            print(f"  • {file}")
        
        print(f"\n🤖 Solution:")
        print("-" * 40)
        print(result['solution'])
        
        if result['research_content'] and "External research unavailable" not in result['research_content']:
            print(f"\n🌐 External Research:")
            print("-" * 40)
            research_lines = result['research_content'].split('\n')[:10]  # 처음 10줄만
            print('\n'.join(research_lines))
            if len(result['research_content'].split('\n')) > 10:
                print("... (more research results available)")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Issue Query & Resolution System")
    parser.add_argument("--issue", required=True, help="Issue description or issue number")
    parser.add_argument("--persist-dir", help="ChromaDB persist directory", 
                       default="./chroma_db")
    
    args = parser.parse_args()
    
    # 환경 변수 설정
    if args.persist_dir:
        os.environ["CHROMA_PERSIST_DIRECTORY"] = args.persist_dir
    
    try:
        # 쿼리 시스템 초기화
        query_system = IssueQuerySystem()
        
        # 이슈 해결
        result = query_system.resolve_issue(args.issue)
        
        # 결과 출력
        query_system.print_resolution_report(result)
        
    except Exception as e:
        print(f"❌ Query system failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()