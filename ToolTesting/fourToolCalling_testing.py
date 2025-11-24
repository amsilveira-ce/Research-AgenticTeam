
from typing import Any, List, Dict, Callable
from langgraph.checkpoint.memory import MemorySaver
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





llama = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7,
)

# ============================================================================
# TOOL 3: PubMed Search (Biomedical and Life Sciences)
# ============================================================================

@tool
def search_pubmed(query: str, max_results: int = 5) -> str:
    """
    Search PubMed for biomedical and life sciences literature.
    Best for: Healthcare, medicine, biology, clinical research
    
    Args:
        query: Medical/biological search query
        max_results: Maximum number of papers
        
    Returns:
        JSON string with paper metadata
    """
    try:
        from Bio import Entrez
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "biopython"])
        from Bio import Entrez
    
    if not query:
        return json.dumps({"error": "Query required", "papers": []})
    
    try:
        # Set email for Entrez (required by NCBI)
        Entrez.email = "research.agent@example.com"
        
        # Search PubMed
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        
        if not id_list:
            return json.dumps({"source": "PubMed", "papers": [], "message": "No papers found"})
        
        # Fetch paper details
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        papers = []
        for article in records['PubmedArticle']:
            medline = article['MedlineCitation']
            article_data = medline['Article']
            
            # Extract abstract
            abstract = ""
            if 'Abstract' in article_data:
                abstract_texts = article_data['Abstract'].get('AbstractText', [])
                abstract = ' '.join([str(text) for text in abstract_texts])
            
            # Extract authors
            authors = []
            if 'AuthorList' in article_data:
                for author in article_data['AuthorList']:
                    if 'LastName' in author and 'Initials' in author:
                        authors.append(f"{author['LastName']} {author['Initials']}")
            
            pmid = str(medline['PMID'])
            
            papers.append({
                "title": article_data.get('ArticleTitle', 'No title'),
                "authors": authors,
                "abstract": abstract or "No abstract available",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "published": str(article_data.get('ArticleDate', 'Unknown')),
                "source": "PubMed",
                "pmid": pmid
            })
        
        return json.dumps({
            "source": "PubMed",
            "papers": papers,
            "count": len(papers)
        })
        
    except Exception as e:
        return json.dumps({"error": str(e), "source": "PubMed", "papers": []})


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
            return json.dumps({"source": "arXiv", "papers": [], "message": "No papers found"})

        papers = []
        for result in results:
            summary = result.summary.replace('\n', ' ')
            authors = [author.name for author in result.authors]
            papers.append({
                "title": result.title,
                "authors": authors,
                "abstract": summary,
                "url": result.entry_id,
                "published": str(result.published),
                "source": "arXiv"
            })

        return json.dumps({
            "source": "arXiv",
            "papers": papers,
            "count": len(papers)
        })

    except Exception as e:
        return json.dumps({"error": str(e), "source": "arXiv", "papers": []})

# =====================================
# Tool Example: CrossRef Search
# =====================================
@tool
def search_crossref(query: str, max_results: int = 5) -> str:
    """Search CrossRef for papers by query."""
    if not query:
        return json.dumps({"error": "Query required", "papers": []})

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
                "journal": item.get("container-title", ["Unknown"])[0],
                "source": "CrossRef"
            })
        
        return json.dumps({
            "source": "CrossRef",
            "papers": papers,
            "count": len(papers)
        })

    except Exception as e:
        return json.dumps({"error": str(e), "source": "CrossRef", "papers": []})


def format_results(result: Dict[str, Any]) -> str:
    """Format agent results into a clean, readable output."""
    messages = result.get("messages", [])
    
    output = []
    output.append("=" * 80)
    output.append("📚 RESEARCH RESULTS")
    output.append("=" * 80)
    
    # Extract tool results and format them
    papers_found = []
    sources_used = set()
    
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                tool_data = json.loads(msg.content)
                if "papers" in tool_data and tool_data["papers"]:
                    papers_found.extend(tool_data["papers"])
                    sources_used.add(tool_data.get("source", "Unknown"))
            except:
                pass
    
    # Show sources used
    if sources_used:
        output.append(f"\n🔍 Sources Queried: {', '.join(sources_used)}")
        output.append(f"📄 Total Papers Retrieved: {len(papers_found)}")
        output.append("-" * 80)
    
    # Format each paper
    if papers_found:
        for i, paper in enumerate(papers_found, 1):
            output.append(f"\n[{i}] {paper.get('title', 'No title')}")
            output.append("-" * 80)
            
            # Authors
            authors = paper.get('authors', [])
            if authors:
                if isinstance(authors, list):
                    author_str = ", ".join(authors[:3])
                    if len(authors) > 3:
                        author_str += f" et al. ({len(authors)} authors)"
                else:
                    author_str = str(authors)
                output.append(f"👥 Authors: {author_str}")
            
            # Publication info
            pub_info = []
            if paper.get('published'):
                pub_info.append(f" {paper['published']}")
            if paper.get('venue'):
                pub_info.append(f" {paper['venue']}")
            if paper.get('journal'):
                pub_info.append(f" {paper['journal']}")
            if paper.get('citation_count') is not None:
                pub_info.append(f" {paper['citation_count']} citations")
            if pub_info:
                output.append(" | ".join(pub_info))
            
            # Abstract preview
            abstract = paper.get('abstract', 'No abstract available')
            if len(abstract) > 300:
                abstract = abstract[:297] + "..."
            output.append(f"\n {abstract}")
            
            # URL
            url = paper.get('url') or paper.get('doi', '')
            if url:
                if not url.startswith('http'):
                    url = f"https://doi.org/{url}"
                output.append(f"\n🔗 {url}")
            
            output.append("")
    
    # Add agent's analysis/summary at the end
    output.append("\n" + "=" * 80)
    output.append(" AGENT ANALYSIS")
    output.append("=" * 80)
    
    # Get the last AI message
    for msg in reversed(messages):
        if hasattr(msg, 'content') and not isinstance(msg, ToolMessage) and not isinstance(msg, SystemMessage):
            if msg.content and len(msg.content) > 50:  # Skip empty or very short messages
                output.append(msg.content)
                break
    
    output.append("\n" + "=" * 80)
    
    return "\n".join(output)


class PaperResearchAgent():
    system_prompt = """
You are a **Multidisciplinary Research Agent** - the primary information retrieval system for academic research.

## Your Role:
You are the MAIN retriever and must provide comprehensive, detailed information. Be thorough and proactive in your analysis.

## Tool Selection Guide:
- **AI/ML/CS/Math/Physics** → search_arXiv
- **Biomedical/Health/Clinical** → search_pubmed  
- **General/Multidisciplinary** → search_semantic_scholar
- **DOI/Citation lookup** → search_crossref

## Your Detailed Analysis Must Include:

### 1. **Query Interpretation & Strategy**
   - Explain how you interpreted the query
   - State which tools you selected and why
   - Mention if you refined the search terms

### 2. **Comprehensive Paper Summaries**
   For EACH relevant paper, provide:
   - **Key Contributions**: What novel insights or methods does this paper introduce?
   - **Methodology**: What approaches, techniques, or frameworks were used?
   - **Findings**: What are the main results and their significance?
   - **Relevance**: How does this paper address the user's query?

### 3. **Cross-Paper Analysis**
   - **Common Themes**: What patterns emerge across multiple papers?
   - **Methodological Trends**: What approaches are gaining traction?
   - **Research Gaps**: What questions remain unanswered?
   - **Contradictions**: Do any papers present conflicting findings?

### 4. **Research Landscape Overview**
   - **Current State**: Where does the field stand on this topic?
   - **Evolution**: How has thinking evolved (if multiple years of papers)?
   - **Leading Groups**: Are there dominant research institutions or authors?
   - **Future Directions**: What are the emerging research questions?

### 5. **Actionable Insights**
   - **Practical Implications**: How can these findings be applied?
   - **Recommendations**: What papers are most critical to read first?
   - **Related Topics**: What adjacent areas should be explored?

## Response Guidelines:
- **BE DETAILED**: This is not a summary service - provide substantial analysis
- **BE SPECIFIC**: Reference specific papers by title when discussing findings
- **BE PROACTIVE**: Anticipate follow-up questions and address them
- **BE COMPREHENSIVE**: Cover methodology, results, implications, and context
- **PROVIDE EVIDENCE**: Base all claims on the retrieved papers
- **HIGHLIGHT QUALITY**: Note highly-cited papers or landmark studies

Your goal is to give the user a complete understanding of the research landscape, not just a list of papers.
"""

    def __init__(self):
        self.tools = [search_crossref, search_arXiv, search_pubmed, search_semantic_scholar]
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
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.checkpoint)

    def invoke(self, query: str, thread_id: str="default") -> dict[str,Any]:
        config = {'configurable': {'thread_id': thread_id}}
        result = self.graph.invoke(
            {
                "messages": [("user", query)]
            },
            config 
        )
        return result 
    
    def invoke_and_format(self, query: str, thread_id: str="default") -> str:
        """Invoke the agent and return formatted results."""
        result = self.invoke(query, thread_id)
        return format_results(result)
    
    def stream(self, query: str, thread_id: str = "default"):
        config = {"configurable": {"thread_id": thread_id}}
        for event in self.graph.stream(
            {"messages": [("user", query)]},
            config,
            stream_mode="values"
        ):
            yield event


if __name__ == "__main__":
    agent = PaperResearchAgent()

    print("\n" + "Test 1: Biomedical Search ".center(80, "="))
    result1 = agent.invoke_and_format("Find recent advances in CRISPR gene editing for cancer therapy", thread_id="session_1")
    print(result1)

    print("\n\n" + "Test 2: AI/ML Search ".center(80, "="))
    result2 = agent.invoke_and_format("Recent transformer architectures for natural language understanding", thread_id="session_2")
    print(result2)

    print("\n\n" + "Test 3: Follow-up Query ".center(80, "="))
    result3 = agent.invoke_and_format("Compare those approaches", thread_id="session_2")
    print(result3)
