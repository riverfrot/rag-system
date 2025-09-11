#!/usr/bin/env python3
"""
RAG System Main CLI Interface
통합 CLI 인터페이스로 ingestion과 query 기능을 제공
"""

import os
import sys
import argparse
from pathlib import Path

# 현재 디렉토리를 Python path에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from rag_pipeline.ingest import RepositoryIngestor
from rag_pipeline.query import IssueQuerySystem

def main():
    parser = argparse.ArgumentParser(
        description="RAG System for Issue Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Repository ingestion
  python main.py ingest --repo https://github.com/riverfrot/sample-spring
  
  # Issue resolution
  python main.py query --issue "ISSUE-2: 데이터 영속성 문제"
  
  # Custom persist directory
  python main.py ingest --repo https://github.com/riverfrot/sample-spring --persist-dir ./custom_db
  python main.py query --issue "Spring Boot database issue" --persist-dir ./custom_db
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest 서브커맨드
    ingest_parser = subparsers.add_parser(
        'ingest', 
        help='Ingest repository into vector database'
    )
    ingest_parser.add_argument(
        '--repo', 
        required=True, 
        help='Repository URL to ingest (e.g., https://github.com/riverfrot/sample-spring)'
    )
    ingest_parser.add_argument(
        '--persist-dir', 
        default='./chroma_db',
        help='ChromaDB persist directory (default: ./chroma_db)'
    )
    
    # Query 서브커맨드
    query_parser = subparsers.add_parser(
        'query', 
        help='Query the system to resolve issues'
    )
    query_parser.add_argument(
        '--issue', 
        required=True,
        help='Issue description (e.g., "ISSUE-2: database persistence problem")'
    )
    query_parser.add_argument(
        '--persist-dir', 
        default='./chroma_db',
        help='ChromaDB persist directory (default: ./chroma_db)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 환경 변수 설정
    os.environ["CHROMA_PERSIST_DIRECTORY"] = args.persist_dir
    
    if args.command == 'ingest':
        print("🚀 Starting Repository Ingestion...")
        print("-" * 50)
        
        try:
            ingestor = RepositoryIngestor()
            ingestor.ingest_repository(args.repo)
            
            print("\n✅ Ingestion completed successfully!")
            print(f"📁 Vector database saved to: {args.persist_dir}")
            print("\n💡 You can now run queries with:")
            print(f"   python main.py query --issue 'your issue description'")
            
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            sys.exit(1)
    
    elif args.command == 'query':
        print("🎯 Starting Issue Resolution...")
        print("-" * 50)
        
        try:
            query_system = IssueQuerySystem()
            result = query_system.resolve_issue(args.issue)
            query_system.print_resolution_report(result)
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            if "vector store" in str(e).lower():
                print("\n💡 Make sure you've run ingestion first:")
                print("   python main.py ingest --repo <repository_url>")
            sys.exit(1)

if __name__ == "__main__":
    main()