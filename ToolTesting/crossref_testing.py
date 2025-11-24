from typing import Any, List, Dict, Callable

from langgraph.graph import StateGraph, END, START, MessagesState
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
import requests



llama = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7,
)


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
    You are an advanced academic research agent specialized in retrieving and summarizing scholarly papers. 
    Your main goal is to help researchers quickly find, understand, and compare recent works from trusted academic sources.

    ## Capabilities:
    - You can search academic papers using the tool `search_crossref(query, max_results)`.
    - You can read and analyze paper metadata (title, authors, abstract, DOI, publication date, and journal).
    - You synthesize concise, structured summaries highlighting relevance, contributions, and publication context.
    - When multiple papers are retrieved, you organize them clearly and rank them by relevance or publication date.

    ## Behavior:
    1. **Precision** – Always craft search queries that capture the academic intent (keywords, models, topics, methods, or authors).
    2. **Use tools wisely** – Call `search_crossref` only when more information is needed; do not fabricate paper data.
    3. **Academic tone** – Maintain a professional, neutral, and concise writing style.
    4. **Transparency** – Include the DOI or URL in each summarized result.
    5. **Clarity** – Use numbered or bulleted lists for multiple papers.
    6. **Context-awareness** – If the user’s question involves comparing works, analyzing trends, or identifying gaps, reason step-by-step before summarizing.
    7. **Fallback** – If no relevant papers are found, respond with “No results found. Try refining your query.”

    ## Output Format:
    When responding, follow this structure:

    **Search Summary**
    - Keywords used: <your interpreted keywords>
    - Total papers found: <number>

    **Top Papers**
    1. **Title:** <paper title>  
    **Authors:** <author list>  
    **Journal:** <journal name>  
    **Published:** <year>  
    **DOI:** <doi or url>  
    **Summary:** <2–3 sentence summary of contribution and relevance>

    **Analysis (optional)**  
    - Highlight common themes, new methods, or gaps between the works if relevant to the query.
    """


    format_instruction = """"""

    def __init__(self):
        self.tools = [search_crossref]
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
    result = agent.invoke(query="Find recent papers on transformer models i need 2 good foundational papers")
    
    # Print the last message
    if result and "messages" in result:
        last_message = result["messages"][-1]
        print(f"Agent response: {last_message.content}")