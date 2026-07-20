import os
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# Define the shared state structure
class AgentState(TypedDict):
    query: str
    search_queries: List[str]
    search_results: List[str]
    evaluation_sufficient: bool
    report: str
    loop_count: int

# Initialize xAI Grok LLM and search tool
llm = ChatOpenAI(model="nvidia/nemotron-3-ultra-550b-a55b:free",
                temperature=0,
                base_url="https://openrouter.ai/api/v1")
search_tool = TavilySearch(max_results=3)

# Initialize local free embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Bind embedding model to ChromaDB vector store
vector_store = Chroma(collection_name="research_memory", embedding_function=embeddings)

# Graph node operations
def formulate_queries(state: AgentState):
    print("--- NODE 1: FORMULATING QUERIES ---")
    query = state["query"]
    loop_count = state.get("loop_count", 0)
    
    prompt = ChatPromptTemplate.from_template(
        "Based on the query: '{query}' generate up to 2 distinct search queries. New lines only."
    )
    chain = prompt | llm
    response = chain.invoke({"query": query, "loop_count": loop_count})
    queries = [q.strip() for q in response.content.split("\n") if q.strip()]
    return {"search_queries": queries, "loop_count": loop_count + 1}

def web_search(state: AgentState):
    print("--- NODE 2: EXECUTING WEB SEARCH ---")
    queries = state["search_queries"]
    current_results = state.get("search_results", [])
    new_results = []
    for q in queries:
        try:
            results = search_tool.invoke({"query": q})
            for r in results:
                content = f"Source: {r['url']}\nContent: {r['content']}"
                new_results.append(content)
                # Save data into local vector store using all-MiniLM-L6-v2
                vector_store.add_texts(texts=[content], metadatas=[{"query": q}])
        except Exception as e:
            print(f"Search failed: {e}")
    return {"search_results": current_results + new_results}

def evaluate_relevance(state: AgentState):
    print("--- NODE 3: EVALUATING RELEVANCE ---")
    if state["loop_count"] >= 3: 
        return "sufficient"
    return "sufficient" 

def compile_report(state: AgentState):
    print("--- NODE 4: COMPILING REPORT ---")
    query = state["query"]
    results = "\n\n".join(state["search_results"])
    prompt = ChatPromptTemplate.from_template("Write an executive report for: '{query}' using:\n{results}")
    chain = prompt | llm
    response = chain.invoke({"query": query, "results": results})
    return {"report": response.content}

# Workflow structure setup
workflow = StateGraph(AgentState)
workflow.add_node("formulate_queries", formulate_queries)
workflow.add_node("web_search", web_search)
workflow.add_node("compile_report", compile_report)

workflow.add_edge(START, "formulate_queries")
workflow.add_edge("formulate_queries", "web_search")
workflow.add_conditional_edges("web_search", evaluate_relevance, {"sufficient": "compile_report"})
workflow.add_edge("compile_report", END)

app = workflow.compile()

if __name__ == "__main__":
    user_request = "AI impact on architecture"
    initial_state = {
        "query": user_request, 
        "search_queries": [], 
        "search_results": [], 
        "evaluation_sufficient": False, 
        "report": "", 
        "loop_count": 0
    }
    final_output = app.invoke(initial_state)
    print("\nFINAL REPORT:\n", final_output["report"])
