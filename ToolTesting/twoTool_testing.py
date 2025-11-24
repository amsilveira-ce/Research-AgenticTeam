from typing import Any, List, Dict, Callable

from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import sys
import subprocess

import json
import requests
import subprocess
import sys


llama = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7,
)
# =====================================
# Tool Example: arXiv Search
# =====================================
@tool
def search_arXiv(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for academic papers related to a given query.

    Args:
        query (str): The topic or keywords to search for.
        max_results (int): Maximum number of papers to retrieve.

    Returns:
        str: A formatted string containing metadata of the retrieved papers.
    """
    try:
        import arxiv
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "arxiv"])
        import arxiv

    if not query or not isinstance(query, str):
        raise ValueError(f"Query must be a non-empty string. Found value: {query}")
    if not isinstance(max_results, int):
        raise ValueError(f"max_results must be an integer. Found value: {max_results}")

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = list(search.results())
        if not results:
            return "No papers found related to the search query."

        paper_details = []
        for i, result in enumerate(results):
            summary = result.summary.replace('\n', ' ')
            authors = ", ".join(author.name for author in result.authors)
            paper_details.append(
                f"Paper {i+1}:\n"
                f"  Title: {result.title}\n"
                f"  Authors: {authors}\n"
                f"  URL: {result.entry_id}\n"
                f"  Abstract: {summary}...\n"
            )

        return "\n".join(paper_details)

    except Exception as e:
        return f"An error occurred while searching arXiv: {str(e)}"

# =====================================
# Tool Example: CrossRef Search
# =====================================
@tool
def search_crossref(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search CrossRef for papers by query."""
    if not query:
        return []

    try:
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": max_results,
            "sort": "relevance",
            "select": "title,author,abstract,DOI,published,container-title,URL"
        }
        headers = {"User-Agent": "ResearchWorkflow/1.0 (mailto:research@example.com)"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        papers = []

        for item in data.get("message", {}).get("items", []):
            authors = [
                f"{a.get('given', '')} {a['family']}" if "given" in a else a["family"]
                for a in item.get("author", [])
            ]
            title = item.get("title", ["No title"])[0]
            abstract = item.get("abstract", "Abstract not available")
            doi = item.get("DOI", "")
            url = item.get("URL", f"https://doi.org/{doi}" if doi else "")
            pub_date = "Unknown"
            for field in ["published", "issued", "created"]:
                if field in item and "date-parts" in item[field]:
                    parts = item[field]["date-parts"][0]
                    if parts:
                        pub_date = "-".join(map(str, parts))
                        break
            papers.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "doi": doi,
                "url": url,
                "published": pub_date,
                "journal": item.get("container-title", ["Unknown"])[0]
            })
        return papers

    except Exception as e:
        print(f"Error in search_crossref: {e}")
        return []

class PaperResearchAgent():

    system_prompt = """
    You are an advanced **Academic Research Agent** specialized in retrieving, analyzing, and summarizing scholarly literature from multiple trusted sources.

    ## Objective
    Your purpose is to assist researchers by finding, comparing, and synthesizing relevant academic papers from reliable databases such as CrossRef and arXiv.

    ## Available Tools
    1. **search_crossref(query, max_results)** — Use this for:
    - Peer-reviewed journal articles
    - Published conference papers
    - Works with DOIs and formal metadata (authors, journal name, year)

    2. **search_arXiv(query, max_results)** — Use this for:
    - Preprints, early-stage research, or trending topics in AI, physics, math, or computer science
    - Cutting-edge or recently posted works not yet in formal journals

    When in doubt, **combine both**: use CrossRef for verified publications and arXiv for the latest findings.

    ## Behavior Guidelines
    - Always interpret the user query carefully before taking action.  
    - If the query is vague, refine it by identifying relevant research fields or key concepts.  
    - Use the tools — do not hallucinate paper metadata.  
    - Combine results intelligently: detect duplicates, prioritize relevance, and provide a balanced view between arXiv (recent) and CrossRef (peer-reviewed).  
    - Maintain an **academic tone**: objective, precise, and concise.  
    - Use markdown formatting for clarity.

    ## Response Structure
    When replying, follow this structure:

    **Search Summary**
    - Keywords interpreted: <your refined keywords>
    - Tools used: <arXiv / CrossRef / both>
    - Total papers found: <number or estimate>

    **Top Papers**
    1. **Title:** <paper title>  
    **Authors:** <author list>  
    **Source:** <arXiv or journal name>  
    **Published:** <year>  
    **DOI/URL:** <link>  
    **Summary:** <2–3 sentences summarizing core contribution and relevance>

    (Continue for each paper)

    **Comparative Analysis (optional)**  
    - Highlight similarities, new directions, and gaps between the retrieved papers.  
    - Mention if some papers are preprints while others are peer-reviewed, and what this means for reliability or trends.

    **Recommendation (optional)**  
    - Suggest next steps, such as narrowing down subtopics, comparing methodologies, or exploring related works.

    ## 🚨 Fallback Rules
    - If no papers are found, respond clearly:  
    “No results found. Try refining your query with specific keywords or fields.”
    - Do **not** invent paper titles, authors, or DOIs.  
    - If tool outputs look incomplete or repetitive, summarize only what’s accurate.

    Be methodical, source-aware, and helpful to academic users.
    """

    format_instruction = """"""

    def __init__(self):
        self.tools = [search_crossref, search_arXiv]
        self.llm_with_tools = llama.bind_tools(self.tools)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph: 

        def agent_node(state: MessagesState):

            messages = state["messages"]

            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=self.system_prompt)] + messages
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
        workflow.add_edge("agent", END)


        return workflow.compile()



    def invoke(self, query: str, thread_id: str="default") -> dict[str,Any]:
            # Use the thread_id for conversation persistence 
            config = {'configurable': {'thread_id': thread_id}}

            result = self.graph.invoke(
                {
                    "messages": [("user", query)]
                },
                config 
            )

            return result 
    
    def stream(self, query: str, thread_id: str = "default"):
        config = {"configurable": {"thread_id": thread_id}}

        for event in self.graph.stream(
            {"messages": [("user", query)]},
            config,
            stream_mode="values"
        ):
            yield event

if __name__ == "__main__":
    # Run this file to teste how the Paper Research Agent works and test new integrations
    agent = PaperResearchAgent()
    result = agent.invoke(query="Find recent papers on transformer models i need 2 good foundational papers and one new rising idea related to it")
    
    # Print the last message
    if result and "messages" in result:
        last_message = result["messages"][-1]
        print(f"Agent response: {last_message.content}")