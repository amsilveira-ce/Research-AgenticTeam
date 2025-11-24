import json
import sys
import subprocess
# from sentence_transformers import util
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from state import ResearchState
from config import Config
from mellea_validation import validate_tool_selection_with_mellea
from tools import all_tools, search_arXiv, search_crossref, search_pubmed, search_semantic_scholar

_ST_UTIL_AVAILABLE = False
try:
    # defer importing util to avoid heavy optional dependency at module import time
    from sentence_transformers import util  # type: ignore
    _ST_UTIL_AVAILABLE = True
except Exception:
    _ST_UTIL_AVAILABLE = False

# ============================================================================
# PLANNER AGENT - Generate Hypotheses
# ============================================================================

def planner_agent(state: ResearchState) -> ResearchState:
    """
    Planner Agent - Analyzes task and generates hypotheses with search keys.
    """
    llm = Config.granite8b
    
    prompt = f'''
    You are a research planner. Given a user task, generate 3 distinct hypotheses.
    For each hypothesis, create 2 search key phrases.

    Generate a JSON response with this structure:
    {{ 
        "hypothesis": [
            {{
                "text": "Hypothesis statement (specific and testable)",
                "search_keys": ["search phrase 1 AND keyword", "search phrase 2 AND keyword"],
                "papers": []
            }}
        ]
    }}

    '''
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state['user_task'])
    ]
    
    response = llm.invoke(messages)
    
    try:
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        result = json.loads(content)
        hypothesis = result['hypothesis']
        
        for hyp in hypothesis:
            if 'papers' not in hyp:
                hyp['papers'] = []
        
        # guard the hypothesis and check the workflow_stage 
        state['hypothesis'] = hypothesis
        state['workflow_stage'] = 'planning_complete'
        
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse planner response: {e}") # Log to terminal
        state['hypothesis'] = []
        
    return state

# ============================================================================
# RESEARCHER AGENT - Autonomous Multi-Tool Research
# ============================================================================

def researcher_agent(state: ResearchState) -> ResearchState:
    """
    Researcher Agent with Tool Autonomy:
    1. Reason about which tools to use
    2. Act by calling tools directly
    3. Collect results
    """
    
    # if there is no hypothesis left to search, then the research is completed and we mark the workflow stage as completed 
    if state["current_hypothesis_index"] >= len(state["hypothesis"]):
        state['workflow_stage'] = 'research_complete'
        return state
    
    # get the current hypothesis that need information retrieval 
    hyp_idx = state["current_hypothesis_index"]
    hypothesis = state['hypothesis'][hyp_idx]
    
    # if this hypothesis have the max amount of papers allowed, continue to the next hypothesis
    if len(hypothesis.get('papers', [])) >= Config.MAX_PAPERS_PER_HYPOTHESIS:
        state["current_hypothesis_index"] += 1
        state['retry_counter'] = 0
        return state
    
    llm_with_tools = Config.llama.bind_tools(all_tools)
    search_keys = hypothesis['search_keys'] 
    
    # STEP 1: Reason about which tools to use
    tool_selection_prompt = f"""
    You are an intelligent research agent. Analyze this hypothesis and determine which research tools are most appropriate.
        **Available Tools:**
        1. search_arXiv - Best for: Computer Science, AI, ML, Physics, Math
        2. search_semantic_scholar - Best for: Cross-disciplinary research, citation analysis
        3. search_pubmed - Best for: Medicine, biology, healthcare, clinical research
        4. search_crossref - Best for: General academic research, any discipline

        Respond in JSON format:
        {{
            "recommended_tools": ["tool_name_1", "tool_name_2"],
            "reasoning": "Why these tools are appropriate"
        }}
    """
    
    selection_messages = [
        SystemMessage(content=tool_selection_prompt),
        HumanMessage(content=f"My hypothesis is {hypothesis['text']}")
    ]
    
    tool_response = llm_with_tools.invoke(selection_messages)
    
    recommended_tools = []

    try:
        content = tool_response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        # we get the tools selected by the agent 
        tool_selection = json.loads(content)
        recommended_tools = tool_selection.get('recommended_tools', [])
    except Exception as e:
        print(f"Researcher Agent ERROR: Error parsing tool selection, using defaults. Error: {e}", file=sys.stderr)
        recommended_tools = ['search_arXiv', 'search_semantic_scholar']
    
    # Validate with Mellea 
    validation = validate_tool_selection_with_mellea(
        state, 
        hypothesis['text'], 
        recommended_tools
    )
    # after calidation we use the validated tools only
    recommended_tools = validation['tools_approved']
    
    # STEP 2: Execute searches 
    all_retrieved_papers = []
    
    if 'tool_usage_stats' not in state:
        state['tool_usage_stats'] = {}
    
    # the tool calling is handleded in the loop - Agents can only write tool calls and not call them!
    for key in search_keys:
        for tool_name in recommended_tools:
            state['tool_usage_stats'][tool_name] = state['tool_usage_stats'].get(tool_name, 0) + 1
            
            try:
                tool_args = {"query": key, "max_results": Config.MAX_SEARCH_RESULTS}
                result = ""

                if tool_name == 'search_arXiv':
                    result = search_arXiv.invoke(tool_args)
                elif tool_name == 'search_semantic_scholar':
                    result = search_semantic_scholar.invoke(tool_args)
                elif tool_name == 'search_pubmed':
                    result = search_pubmed.invoke(tool_args)
                elif tool_name == 'search_crossref':
                    result = search_crossref.invoke(tool_args)
                else:
                    result = json.dumps({"error": "Unknown tool", "papers": []})

                try:
                    result_data = json.loads(result)
                    papers = result_data.get('papers', [])
                    if papers:
                        all_retrieved_papers.extend(papers)
                    elif result_data.get('error'):
                        print(f"Researcher Agent ERROR: Tool {tool_name} returned an error: {result_data['error']}", file=sys.stderr)
                except json.JSONDecodeError:
                    # Log the specific JSON parsing error
                    print(f"RESEARCHER: Failed to parse JSON from {tool_name}. Error: {e}. Result was: {result[:200]}...", file=sys.stderr)
                        
            except Exception as e:
                # Log any other error during tool execution (e.g., API connection error)
                print(f"RESEARCHER: Critical error calling tool {tool_name}. Error: {e}", file=sys.stderr)
    
    # update the retrieved papers for the current hypothesis 
    state['current_retrieved_papers'] = all_retrieved_papers
    return state

# ============================================================================
# CURATOR AGENT - Semantic Evaluation and Selection
# ============================================================================

def curator_agent(state: ResearchState) -> ResearchState:
    """
    Curator Agent: Evaluates papers using semantic similarity and LLM reasoning.
    """

    # the curator needs to validate the current hypothesis
    hyp_idx = state["current_hypothesis_index"]
    if hyp_idx >= len(state["hypothesis"]):
        return state
    
    hypothesis = state['hypothesis'][hyp_idx]
    retrieved_papers = state.get('current_retrieved_papers', [])
    
    # If we dont get retrieved papers, move to the next hypothesis
    # Attention !!! not a good implementation for reliable workflow 
    # but handle infinite loops 
    if not retrieved_papers:
        # No papers found, move to next hypothesis
        state["current_hypothesis_index"] += 1
        state['retry_counter'] = 0
        state['current_retrieved_papers'] = []
        return state
    

    # Remove possible duplicates by title - different tools might give same results 
    # Attention !!! do we really need this?
    unique_papers = []
    seen_titles = set()
    for paper in retrieved_papers:
        title = paper.get('title', '').lower().strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_papers.append(paper)
    

    # STEP 1: Semantic Similarity Scoring
    hypothesis_text = hypothesis['text']
    # lazy-load the embedding model
    embedder = Config.get_embedding_model()
    hypothesis_embedding = embedder.encode(hypothesis_text, convert_to_tensor=True)
    
    papers_with_scores = []
    for paper in unique_papers:
        # Attention !! we are only using the abstract for this, maybe use the full paper for validation can generate better results 
        paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}" # get paper content 
        paper_embedding = embedder.encode(paper_text, convert_to_tensor=True)  # generate embedding vector 

        # compute similarity; prefer sentence_transformers.util if available
        if _ST_UTIL_AVAILABLE:
            from sentence_transformers import util as _st_util
            similarity_score = _st_util.cos_sim(hypothesis_embedding, paper_embedding).item()
        else:
            # Fallback: cosine via numpy to avoid hard dependency
            try:
                import numpy as _np

                def _cosine(a, b):
                    a = _np.asarray(a)
                    b = _np.asarray(b)
                    if a.ndim == 1:
                        a = a.reshape(1, -1)
                    if b.ndim == 1:
                        b = b.reshape(1, -1)
                    a_norm = _np.linalg.norm(a, axis=1)
                    b_norm = _np.linalg.norm(b, axis=1)
                    denom = a_norm * b_norm
                    # avoid divide by zero
                    denom = _np.where(denom == 0, 1e-8, denom)
                    return (_np.sum(a * b, axis=1) / denom)[0]

                similarity_score = float(_cosine(hypothesis_embedding, paper_embedding))
            except Exception:
                # Last resort: 0 similarity
                similarity_score = 0.0

        
        paper_copy = paper.copy()

        # add the score to the current paper - will be used for validation
        paper_copy['semantic_similarity'] = similarity_score
        papers_with_scores.append(paper_copy)
    
    # sort the papers based on the scores 
    papers_with_scores.sort(key=lambda x: x['semantic_similarity'], reverse=True)
    
    # STEP 2: LLM-based Relevance Evaluation
    # Attention !!! might be a waste of llm usage 
    top_candidates = papers_with_scores[:min(8, len(papers_with_scores))]
    
    llm = Config.granite8b
    
    candidates_summary = []
    for i, paper in enumerate(top_candidates):

        candidates_summary.append(f"""
        Paper {i}:
        Title: {paper['title']}
        Abstract: {paper['abstract'][:250]}...
        Semantic Similarity: {paper['semantic_similarity']:.3f}
        """)
        
        evaluation_prompt = f"""You are a research curator.
        **Hypothesis:** {hypothesis_text}
        **Candidate Papers:**
        {''.join(candidates_summary)}
        """
    
    mock_human_task = f''' **Task:**
    Select the TOP {Config.MAX_PAPERS_PER_HYPOTHESIS} papers that best match the hypothesis.

    Return ONLY a JSON response:
    {{
        "selected_indices": [0, 2],
        "reasoning": "Detailed explanation of why these papers are most relevant"
    }}
    Indices should be 0-based. Select exactly {Config.MAX_PAPERS_PER_HYPOTHESIS} papers.
    '''
    messages = [
        SystemMessage(content=evaluation_prompt),
        HumanMessage(content=mock_human_task)
    ]
    
    response = llm.invoke(messages)
    
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        selection = json.loads(content)
        selected_indices = selection.get('selected_indices', [])

        # try to get the reasoning behind the choices that the llm did 
        reasoning = selection.get('reasoning', 'No reasoning provided')
        
        selected_papers = []
        for idx in selected_indices[:Config.MAX_PAPERS_PER_HYPOTHESIS]:
            if 0 <= idx < len(top_candidates):
                paper = top_candidates[idx].copy()
                paper['selection_reasoning'] = reasoning
                selected_papers.append(paper)
        
        if len(selected_papers) >= Config.MAX_PAPERS_PER_HYPOTHESIS:
            hypothesis['papers'].extend(selected_papers[:Config.MAX_PAPERS_PER_HYPOTHESIS])
            state['all_papers'].extend(selected_papers[:Config.MAX_PAPERS_PER_HYPOTHESIS])
        else:
            state['retry_counter'] += 1
    
    except (json.JSONDecodeError, KeyError) as e:
        # Fallback: use top papers by semantic similarity
        if len(papers_with_scores) >= Config.MAX_PAPERS_PER_HYPOTHESIS:
            fallback_papers = papers_with_scores[:Config.MAX_PAPERS_PER_HYPOTHESIS]
            for paper in fallback_papers:
                paper['selection_reasoning'] = "Selected by semantic similarity (fallback)"
            
            hypothesis['papers'].extend(fallback_papers)
            state['all_papers'].extend(fallback_papers)
        else:
            state['retry_counter'] += 1
    
    # Check max retries
    if state['retry_counter'] >= state['max_retries']:
        state["current_hypothesis_index"] += 1
        state['retry_counter'] = 0
    elif len(hypothesis['papers']) >= Config.MAX_PAPERS_PER_HYPOTHESIS:
        state["current_hypothesis_index"] += 1
        state['retry_counter'] = 0

    state['current_retrieved_papers'] = []
    return state

# ============================================================================
# REPORT GENERATOR - Create Final Literature Review
# ============================================================================

def report_generator(state: ResearchState) -> ResearchState:
    """
    Generate a comprehensive literature review report from all findings.
    """
    llm = Config.granite8b
    
    research_summary = []
    for i, hyp in enumerate(state['hypothesis'], 1):
        papers = hyp.get('papers', [])
        research_summary.append(f"""
        ### Hypothesis {i}: {hyp['text']}

        **Papers Found:** {len(papers)}
        """)
        if papers:
            research_summary.append("**Key Findings:**")
            for j, paper in enumerate(papers, 1):
                research_summary.append(f"""
                {j}. **{paper['title']}**
                - Authors: {', '.join(paper.get('authors', ['Unknown'])[:3])}
                - Source: {paper.get('source', 'Unknown')}
                - Relevance Score: {paper.get('semantic_similarity', 0):.3f}
                - URL: {paper.get('url', 'N/A')}
                - *Abstract:* {paper.get('abstract', 'N/A')[:200]}...
                """)
        else:
            research_summary.append("\n*No relevant papers were selected for this hypothesis.*\n")

    
    summary_text = '\n'.join(research_summary)
    
    report_prompt = f"""You are a research analyst. Create a comprehensive literature review report.
        **Original Task:** {state['user_task']}

        **Research Summary:**
        {summary_text}

        **Tool Usage Statistics:**
        {json.dumps(state.get('tool_usage_stats', {}), indent=2)}

        Create a well-structured report in Markdown that:
        1. Provides a brief executive summary of the findings.
        2. Presents the findings for each hypothesis.
        3. Includes a list of all selected papers with their details.
        4. Concludes with potential themes or areas for further research.
    """
    
    messages = [
        SystemMessage(content="You are an expert research analyst."),
        HumanMessage(content=report_prompt)
    ]
    
    response = llm.invoke(messages)
    state['final_report'] = response.content
    
    return state