from typing import Any, List, Dict
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
import json
import subprocess
import sys


@tool
def search_pubmed(query: str, max_results: int = 5) -> str:
    """
    Search PubMed for biomedical and life sciences literature.
    Returns JSON metadata for top results.
    """
    try:
        from Bio import Entrez
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "biopython"])
        from Bio import Entrez

    if not query:
        return json.dumps({"error": "Query required", "papers": []})

    try:
        Entrez.email = "research.agent@example.com"
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        ids = record["IdList"]
        if not ids:
            return json.dumps({"source": "PubMed", "papers": [], "message": "No papers found"})

        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        papers = []
        for article in records['PubmedArticle']:
            medline = article['MedlineCitation']
            art = medline['Article']
            abstract = " ".join(art['Abstract'].get('AbstractText', [])) if 'Abstract' in art else "No abstract"
            authors = [
                f"{a['LastName']} {a['Initials']}"
                for a in art.get('AuthorList', [])
                if 'LastName' in a and 'Initials' in a
            ]
            pmid = str(medline['PMID'])
            papers.append({
                "title": art.get('ArticleTitle', 'No title'),
                "authors": authors,
                "abstract": abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source": "PubMed",
                "pmid": pmid
            })
        return json.dumps({"source": "PubMed", "papers": papers, "count": len(papers)})

    except Exception as e:
        return json.dumps({"error": str(e), "source": "PubMed", "papers": []})


llama = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7,
)


class PaperResearchAgent:
    system_prompt = """
    You are a biomedical research agent specialized in discovering and summarizing scholarly papers using the PubMed database.
    You only have access to one tool: **search_pubmed(query, max_results)**.

    Your task is to find and summarize the most relevant, evidence-based research papers related to the user's query.

    ## GOAL
    Provide clear, clinically-minded, and citation-rich overviews of biomedical topics using PubMed as your only source.

    ## REASONING & ACTION FRAMEWORK (ReAct)

    ### Step 1 — Thought
    Analyze the user's query.
    - Identify the biomedical domain (e.g., oncology, cardiology, genetics, infectious disease).
    - Determine the user's intent (e.g., find recent clinical trials, summarize mechanisms, compare treatments, identify biomarkers).
    - Expand abbreviations and acronyms (e.g., “BP” → “blood pressure”; “NSCLC” → “non-small cell lung cancer”).
    - When appropriate, translate lay terms to clinical/MeSH terms to improve retrieval (e.g., “heart attack” → “myocardial infarction”).

    If the query is vague, refine it by adding synonyms, MeSH terms, or narrower population/intervention/outcome keywords.

    ### Step 2 — Action
    Call **search_pubmed(query, max_results)** with the refined query.
    - Only use this tool; do not call any other tool or invent results.
    - Prefer searches that include clinical filters when the user asks about trials or treatment efficacy (e.g., keywords like “randomized”, “clinical trial”, “systematic review”).

    ### Step 3 — Observation
    Carefully examine the tool output.
    - Extract title, authors, journal, year, PMID/URL, and abstract or summary.
    - Note study type (e.g., randomized controlled trial, cohort study, review), sample size if available, primary outcome, and key conclusions.
    - If the tool returns structured JSON, parse it and summarize the key fields.

    ### Step 4 — Final Answer
    Compose a concise, evidence-focused summary including:
    - A short contextual overview of what the retrieved papers collectively indicate.
    - 3–5 structured entries summarizing the top papers (title, authors, journal, year, PMID/URL, 2–3 sentence summary of main findings).
    - Clinical implications or limitations, and any notable gaps or conflicts in the evidence.
    - If applicable, recommend next steps (e.g., narrower search terms, review articles, or clinical guidelines).

    ## RESPONSE FORMAT

    **Search Summary**
    - Query interpreted: <refined keywords / MeSH terms>
    - Total papers found: <number>
    - Data source: PubMed

    **Top Papers**
    1. **Title:** <paper title>  
    **Authors:** <authors>  
    **Journal:** <journal>  
    **Year:** <year>  
    **PMID / URL:** <link>  
    **Summary:** <2–3 sentences summarizing study design, primary outcome, and main conclusion>

    **Clinical Insights**
    - <Key findings, applicability, and limitations>
    - <Potential next steps or related searches>

    ## RULES
    - Only use the **search_pubmed** tool.
    - Do **not** fabricate papers, PMIDs, or results.
    - Maintain a clinical and evidence-based tone; avoid speculative statements.
    - If no relevant papers are found, say: “No relevant PubMed papers found. Try refining the search using MeSH terms or more specific clinical keywords.”
    - Always include PMIDs or PubMed URLs for transparency.
    - If a study is a preclinical or animal study, clearly label it as such.
    - If clinical practice or patient care is discussed, include an explicit disclaimer: this is a literature summary and not medical advice.

    Your reasoning and output should match the clarity and rigor expected from a biomedical research assistant.
    """


    def __init__(self):
        self.tools = [search_pubmed]
        self.llm_with_tools = llama.bind_tools(self.tools)
        self.checkpoint = MemorySaver()
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
        workflow.add_conditional_edges(
            "agent",
            lambda x: "tools" if "tool_calls" in str(x["messages"][-1]) else END
        )
        workflow.add_edge("tools", "agent")

        # ✅ Compile with memory checkpoint
        return workflow.compile(checkpointer=self.checkpoint)


    def invoke(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke({"messages": [("user", query)]}, config)
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
    # First query
    result = agent.invoke("Find 4 studies about telomerase in skin cancer.")
    print("Agent response (1):", result["messages"][-1].content)
