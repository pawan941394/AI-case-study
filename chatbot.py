
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.tools.crawl4ai import Crawl4aiTools
from agno.tools import tool
from rag_tool import RAGTool
import os
from dotenv import load_dotenv

load_dotenv()
# API Key
API_KEY = os.getenv('OPENAI_API_KEY')

# Cache for RAG instances (so we don't reload same PDFs)
_rag_cache = {}

@tool
def answer_from_pdf(pdf_path: str, query: str, top_k: int = 3) -> str:
    """
    Answer questions based on a PDF document using RAG (Retrieval Augmented Generation).
    Loads the PDF, searches for relevant content, and generates an AI-powered answer.
    
    Args:
        pdf_path: Full path to the PDF file (e.g., C:\\Users\\...\\resume.pdf)
        query: The question to answer about the PDF content
        top_k: Number of relevant chunks to use for context (default: 3)
    
    Returns:
        AI-generated answer based on PDF content
    """
    if pdf_path not in _rag_cache:
        print(f"\n🔧 [RAG] Initializing for: {pdf_path}")
        rag = RAGTool(api_key=API_KEY)
        rag.process_pdf(pdf_path)
        _rag_cache[pdf_path] = rag
    else:
        print(f"\n✅ [RAG] Using cached version")
    
    rag = _rag_cache[pdf_path]
    answer = rag.answer_query(query, top_k=top_k, model="gpt-4o-mini")
    return answer

@tool
def search_pdf_content(pdf_path: str, query: str, top_k: int = 3) -> str:
    """
    Search a PDF document for relevant content without generating an answer.
    Returns the most similar text chunks from the document.
    
    Args:
        pdf_path: Full path to the PDF file
        query: The search query
        top_k: Number of top results to return (default: 3)
    
    Returns:
        Relevant text chunks from the PDF with similarity scores
    """
    if pdf_path not in _rag_cache:
        print(f"\n🔧 [RAG] Initializing for: {pdf_path}")
        rag = RAGTool(api_key=API_KEY)
        rag.process_pdf(pdf_path)
        _rag_cache[pdf_path] = rag
    else:
        print(f"\n✅ [RAG] Using cached version")
    
    rag = _rag_cache[pdf_path]
    results = rag.search(query, top_k=top_k)
    
    output = []
    for i, (chunk, score) in enumerate(results, 1):
        output.append(f"**Result {i}** (Score: {score:.3f})\n{chunk[:300]}...")
    
    return "\n\n".join(output)




db = SqliteDb(db_file="tmp/agents.db")


agent = Agent(
    model=OpenAIChat(id="gpt-4o", api_key=API_KEY, max_completion_tokens=200),
    db=db,
    tools=[Crawl4aiTools(), answer_from_pdf, search_pdf_content],
    add_history_to_context=True,
    enable_agentic_memory=True,
    markdown=True,
    # learning=True,
)

user_id = "user@example.com"
session_id = "main_session1"




# # Interactive Chat Loop
# while True:
#     try:
#         user_query = input("\n💬 You: ").strip()
        
#         if not user_query:
#             continue
            
#         if user_query.lower() in ['exit', 'quit', 'q']:
#             print("\n👋 Goodbye!")
#             break
        
#         print("\n🤖 Agent:")
#         print("-" * 70)
        
#         agent.print_response(
#             user_query,
#             user_id=user_id,
#             session_id=session_id,
#             stream=True,
#         )
        
#         print("\n" + "=" * 70)
        
#     except KeyboardInterrupt:
#         print("\n\n👋 Goodbye!")
#         break
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         print("=" * 70)
