import os
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    class HuggingFaceEmbeddings:
        def __init__(self, *args, **kwargs):
            self.model_name = kwargs.get("model_name", "")

try:
    from langchain_tavily import TavilySearch
except ImportError:
    class TavilySearch:
        def __init__(self, *args, **kwargs):
            self.max_results = kwargs.get("max_results", 3)

        def invoke(self, query):
            return []

try:
    from langchain_chroma import Chroma
except ImportError:
    class Chroma:
        def __init__(self, *args, **kwargs):
            self._texts = []

        def add_texts(self, texts, metadatas=None):
            self._texts.extend(texts)
            return []

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

# Initialize LLM via OpenRouter using a stable free multi-language model
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy"
llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Core Tool Fix: Ensure search_tool is always explicitly instantiated
search_tool = TavilySearch(max_results=3)


def safe_llm_invoke(prompt, input_data, fallback_text: str | None = None):
    try:
        # Fixed invoke path to safely stringify without internal Pydantic serializations
        formatted_prompt = prompt.format(**input_data)
        response = llm.invoke(formatted_prompt)
        return getattr(response, "content", str(response))
    except Exception as exc:
        print(f"LLM call failed: {exc}")
        return fallback_text

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
        "Based on the query: '{query}' generate up to 2 distinct search queries. "
        "The output search queries MUST be written in the same language as the user query. New lines only."
    )
    response_text = safe_llm_invoke(
        prompt,
        {"query": query, "loop_count": loop_count},
        fallback_text=query,
    )
    if response_text:
        queries = [q.strip() for q in str(response_text).split("\n") if q.strip()]
    else:
        queries = [query]
    return {"search_queries": queries, "loop_count": loop_count + 1}

def web_search(state: AgentState):
    print("--- NODE 2: EXECUTING WEB SEARCH ---")
    queries = state["search_queries"]
    current_results = state.get("search_results", [])
    new_results = []
    
    for q in queries:
        try:
            results = search_tool.invoke({"query": q})
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and 'url' in r and 'content' in r:
                        content = f"Source: {r['url']}\nContent: {r['content']}"
                        new_results.append(content)
                        vector_store.add_texts(texts=[content], metadatas=[{"query": q}])
                    else:
                        content = f"Result: {r}"
                        new_results.append(content)
            else:
                content = f"Result: {results}"
                new_results.append(content)
        except Exception as e:
            print(f"Search failed: {e}")
            
    return {"search_results": current_results + new_results}

def evaluate_relevance(state: AgentState):
    print("--- NODE 3: EVALUATING RELEVANCE (CRITIC AGENT) ---")
    
    if state["loop_count"] >= 3: 
        print("-> Loop limit hit. Routing straight to final report.")
        return "sufficient"
        
    query = state["query"]
    results = "\n\n".join(state["search_results"])
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert Reviewer/Critic Agent. Evaluate if the following gathered web results provide "
        "sufficient and detailed information to deeply answer this query: '{query}'.\n\n"
        "Gathered Knowledge Context:\n{results}\n\n"
        "Decision Criteria: Output exactly one lowercase word. If the data is fully enough to compose an "
        "executive report, output 'sufficient'. If it lacks depth or key details, output 'insufficient'."
    )
    response_text = safe_llm_invoke(
        prompt,
        {"query": query, "results": results},
        fallback_text="sufficient" if results else "insufficient",
    )
    decision = str(response_text or "sufficient").strip().lower()
    
    print(f"-> Critic Agent Decision: {decision}")
    if "insufficient" in decision:
        return "insufficient"
    return "sufficient" 

def compile_report(state: AgentState):
    print("--- NODE 4: COMPILING REPORT ---")
    query = state["query"]
    results = "\n\n".join(state["search_results"])
    
    prompt = ChatPromptTemplate.from_template(
        "Write an exhaustive, high-quality executive report for: '{query}' using the provided context:\n{results}\n\n"
        "CRITICAL RULE: The final report MUST be written completely in the SAME LANGUAGE as the query '{query}'."
    )
    response_text = safe_llm_invoke(
        prompt,
        {"query": query, "results": results},
        fallback_text=f"Executive report for: {query}\n\nNo external context was available, so this is a concise fallback report.\n",
    )
    return {"report": response_text or f"Executive report for: {query}"}

# Workflow structure setup
workflow = StateGraph(AgentState)
workflow.add_node("formulate_queries", formulate_queries)
workflow.add_node("web_search", web_search)
workflow.add_node("compile_report", compile_report)

workflow.add_edge(START, "formulate_queries")
workflow.add_edge("formulate_queries", "web_search")

workflow.add_conditional_edges(
    "web_search", 
    evaluate_relevance, 
    {
        "sufficient": "compile_report",
        "insufficient": "formulate_queries"
    }
)
workflow.add_edge("compile_report", END)

app = workflow.compile()

if __name__ == "__main__":
    user_request = "أثر الذكاء الاصطناعي على الهندسة المعمارية"
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
