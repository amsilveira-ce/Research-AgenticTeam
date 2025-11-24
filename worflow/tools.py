from langchain_core.tools import tool
import json
import sys
import subprocess
import requests

try:
    import arxiv
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "arxiv"])
    import arxiv

try:
    from Bio import Entrez
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "biopython"])
    from Bio import Entrez


# ============================================================================
# TOOL 1: arXiv Search
# ============================================================================

@tool
def search_arXiv(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for academic papers in CS, Physics, Math, and related fields.
    Best for: AI, ML, theoretical computer science, physics
    """
    if not query or not isinstance(query, str):
        return json.dumps({"error": "Query must be a non-empty string", "papers": []})
    
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
            papers.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary.replace('\n', ' '),
                "url": result.entry_id,
                "published": str(result.published),
                "source": "arXiv",
                "categories": result.categories
            })

        return json.dumps({"source": "arXiv", "papers": papers, "count": len(papers)})

    except Exception as e:
        return json.dumps({"error": str(e), "source": "arXiv", "papers": []})

# ============================================================================
# TOOL 2: Semantic Scholar Search
# ============================================================================

@tool
def search_semantic_scholar(query: str, max_results: int = 5) -> str:
    """
    Search Semantic Scholar - comprehensive academic search with citation data.
    Best for: Cross-disciplinary research, citation analysis, paper influence
    """
    if not query:
        return json.dumps({"error": "Query required", "papers": []})
    
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,citationCount,url,venue,publicationDate"
        }
        
        response = requests.get(url, params=params, timeout=10)
        
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

# ============================================================================
# TOOL 3: PubMed Search
# ============================================================================

@tool
def search_pubmed(query: str, max_results: int = 5) -> str:
    """
    Search PubMed for biomedical and life sciences literature.
    Best for: Healthcare, medicine, biology, clinical research
    """
    if not query:
        return json.dumps({"error": "Query required", "papers": []})
    
    try:
        Entrez.email = "research.agent@example.com"
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        
        if not id_list:
            return json.dumps({"source": "PubMed", "papers": [], "message": "No papers found"})
        
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        
        papers = []
        for article in records['PubmedArticle']:
            medline = article['MedlineCitation']
            article_data = medline['Article']
            
            abstract = ""
            if 'Abstract' in article_data:
                abstract_texts = article_data['Abstract'].get('AbstractText', [])
                abstract = ' '.join([str(text) for text in abstract_texts])
            
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

# ============================================================================
# TOOL 4: CrossRef Search
# ============================================================================

@tool
def search_crossref(query: str, max_results: int = 5) -> str:
    """
    Search CrossRef for academic papers across all disciplines via DOI metadata.
    Best for: General research, finding papers with DOIs, journal articles
    """
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
        
        if response.status_code == 200:
            data = response.json()
            papers = []
            
            for item in data.get("message", {}).get("items", []):
                authors = []
                for author in item.get("author", []):
                    if "family" in author:
                        name = author.get("family", "")
                        if "given" in author:
                            name = f"{author['given']} {name}"
                        authors.append(name)
                
                title_list = item.get("title", ["No title"])
                title = title_list[0] if title_list else "No title"
                
                abstract = item.get("abstract", "Abstract not available via CrossRef")
                doi = item.get("DOI", "")
                url = item.get("URL", f"https://doi.org/{doi}" if doi else "")
                
                pub_date = "Unknown"
                if "published" in item:
                    date_parts = item["published"].get("date-parts", [[]])[0]
                    if date_parts:
                        pub_date = "-".join(map(str, date_parts))
                
                papers.append({
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "url": url,
                    "doi": doi,
                    "published": pub_date,
                    "journal": item.get("container-title", ["Unknown"])[0],
                    "source": "CrossRef"
                })
            
            return json.dumps({
                "source": "CrossRef",
                "papers": papers,
                "count": len(papers)
            })
        else:
            return json.dumps({
                "error": f"API returned status {response.status_code}",
                "source": "CrossRef",
                "papers": []
            })
            
    except Exception as e:
        return json.dumps({"error": str(e), "source": "CrossRef", "papers": []})

# Tool list exportation ready
all_tools = [search_arXiv, search_semantic_scholar, search_pubmed, search_crossref]