from typing import TypedDict, List, Dict, Any, Optional

class ResearchState(TypedDict):
    """State shared across all agents in the workflow"""
    
    user_task: str 
    hypothesis: List[Dict[str, Any]]  # {text, search_keys, papers[], validation}
    all_papers: List[Dict[str, Any]]  # Global paper repository
    current_hypothesis_index: int
    retry_counter: int 
    max_retries: int 
    messages: List[Any]
    current_retrieved_papers: List[Dict[str, Any]]  # Temporary storage
    validation_results: List[Dict[str, Any]]  # Mellea validation results
    tool_usage_stats: Dict[str, int]  # Track which tools are used
    final_report: str
    workflow_stage: str  # Track current stage