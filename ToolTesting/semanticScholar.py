from typing import Any, List, Dict, Callable

from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
import sys
import subprocess
import json 
from sentence_transformers import SentenceTransformer, util 
import numpy as np 


import json
import requests
import subprocess
import sys


# =====================================
# Tool: Semantic Scholar 
# =====================================
@tool
def search_semantic_scholar(query: str, max_results: int = 5) -> str:
    """
    Search Semantic Scholar - comprehensive academic search with citation data.
    Best for: Cross-disciplinary research, citation analysis, paper influence
    
    Args:
        query: Search query
        max_results: Maximum number of papers
        
    Returns:
        JSON string with paper metadata including citations
    """
    try:
        import requests
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests
    
    if not query:
        return json.dumps({"error": "Query required", "papers": []})
    
    try:
        # Semantic Scholar API endpoint
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,citationCount,url,venue,publicationDate"
        }
        headers = {
            "User-Agent": "ResearcherAgent/1.0 (mailto:ResearcherAgent@example.com)"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=100)

        
        if response.status_code == 200:
            data = response.json()
            papers = []
            
            for paper in data.get("data", []):
                if paper:
                    papers.append({
                        "title": paper.get("title", "No title"),
                        "authors": [a.get("name", "Unknown") for a in paper.get("authors", [])],
                        "abstract": paper.get("abstract", "No abstract available"),
                        "url": paper.get("url", ""),
                        "published": paper.get("publicationDate", paper.get("year", "Unknown")),
                        "citation_count": paper.get("citationCount", 0),
                        "venue": paper.get("venue", "Unknown"),
                        "source": "Semantic Scholar"
                    })
            
            return json.dumps({
                "source": "Semantic Scholar",
                "papers": papers,
                "count": len(papers)
            })
        else:
            return json.dumps({
                "error": f"API returned status {response.status_code}",
                "source": "Semantic Scholar",
                "papers": []
            })
            
    except Exception as e:
        return json.dumps({"error": str(e), "source": "Semantic Scholar", "papers": []})


llama = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7,
)

class PaperResearchAgent():
    system_prompt = """
        You are an academic research agent specialized in discovering and summarizing scholarly papers using the Semantic Scholar database. 
        You only have access to one tool: **search_semantic_scholar(query, max_results)**.

        Your task is to find and summarize the most relevant, high-quality research papers related to the user's query.

        ---

        ## 🎯 GOAL
        Provide clear, insightful, and citation-rich overviews of academic topics using Semantic Scholar as your only source.

        ---

        ## 🧠 REASONING & ACTION FRAMEWORK (ReAct)

        ### Step 1 — Thought
        Analyze the user's query.  
        Infer the **research domain**, **intent**, and **keywords** to build a focused search query.  
        If the query is vague, refine it using synonyms or related concepts to ensure relevance.

        ### Step 2 — Action
        Use **search_semantic_scholar(query, max_results)** to retrieve relevant papers.  
        Do not use any other tools or make up results.

        ### Step 3 — Observation
        Examine the tool output carefully. Identify:
        - Titles, authors, and publication year
        - Abstract or summary of each paper
        - Citation relevance or novelty if mentioned

        ### Step 4 — Final Answer
        Compose a concise academic summary that includes:
        - A short contextual overview of what the papers collectively reveal
        - 3–5 structured entries summarizing top papers
        - Observations about emerging trends or notable gaps

        ---

        ## 🧾 RESPONSE FORMAT

        **Search Summary**
        - Query interpreted: <refined keywords or focus>
        - Total papers found: <number>
        - Data source: Semantic Scholar

        **Top Papers**
        1. **Title:** <paper title>  
        **Authors:** <authors>  
        **Published:** <year>  
        **URL:** <paper link>  
        **Summary:** <2–3 sentences summarizing the paper's core contribution>

        **Insights**
        - <Highlight key findings, emerging methods, or knowledge gaps>

        ---

        ## ⚙️ RULES
        - Only use the **search_semantic_scholar** tool.
        - Never fabricate data or papers.
        - Maintain an academic, objective, and concise tone.
        - If no papers are found, say: “No relevant papers found. Try refining your query.”
        - Always include URLs for transparency.

        Your reasoning and writing must reflect the clarity of a professional academic researcher.
        """


    def __init__(self):
        self.tools = [search_semantic_scholar]
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