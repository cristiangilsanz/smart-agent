import os

from dotenv import load_dotenv

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool

from supabase.client import Client, create_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
system_prompt_path = os.path.join(BASE_DIR, "config", "system_prompt.txt")

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The product of a and b.
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The sum of a and b.
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The difference of a and b (a - b).
    """
    return a - b


@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.

    Args:
        a: Numerator integer.
        b: Denominator integer.

    Returns:
        The quotient of a divided by b.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:
    """Calculate the modulus (remainder) of two numbers.

    Args:
        a: Dividend integer.
        b: Divisor integer.

    Returns:
        The remainder of a divided by b.
    """
    return a % b


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return up to 2 documents.

    Args:
        query: The search query string.

    Returns:
        A dictionary with key 'wiki_results' containing formatted search results.
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}


@tool
def web_search(query: str) -> str:
    """Search the web via Tavily and return up to 3 results.

    Args:
        query: The search query string.

    Returns:
        A dictionary with key 'web_results' containing formatted search results.
    """
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}


@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for academic papers and return up to 3 results.

    Args:
        query: The search query string.

    Returns:
        A dictionary with key 'arvix_results' containing formatted search results (first 1000 characters).
    """
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}

with open(system_prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()

sys_msg = SystemMessage(content=system_prompt)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"), 
    os.environ.get("SUPABASE_SERVICE_KEY"))
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents_langchain",
)

create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)

tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    web_search,
    arvix_search,
]


def build_graph(provider: str = "groq"):
    if provider == "groq":
        llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0,
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")
    
    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def retriever(state: MessagesState):
        similar_question = vector_store.similarity_search(state["messages"][0].content)
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()


class BasicAgent:
    def __init__(self, provider: str = "groq"):
        print("BasicAgent initialized.")
        self.graph = build_graph(provider=provider)

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        messages = [HumanMessage(content=question)]
        result = self.graph.invoke({"messages": messages})
        answer = result['messages'][-1].content
        return answer[14:]


if __name__ == "__main__":
    agent = BasicAgent(provider="groq")
    question = (
        "What is the capital of France?"
    )
    answer = agent(question)
    print(answer)
