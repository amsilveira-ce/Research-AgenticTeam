import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


import streamlit as st
from src.workflow import build_workflow
from src.state import ResearchState
from src.config import MAX_RETRIES, MAX_PAPERS_PER_HYPOTHESIS
import time
import json


# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Agent Research Workflow",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.title("🤖 Research Agent Control")
st.sidebar.markdown("""
This app uses a team of AI agents (Planner, Researcher, Curator) 
to perform a literature review based on your task.
""")

st.sidebar.header("Workflow Configuration")
max_papers_per_hyp = st.sidebar.slider(
    "Max Papers per Hypothesis", 
    min_value=1, 
    max_value=5, 
    value=MAX_PAPERS_PER_HYPOTHESIS,
    help="The number of papers the Curator agent will select for each hypothesis."
)

max_retries = st.sidebar.slider(
    "Max Retries", 
    min_value=1, 
    max_value=5, 
    value=MAX_RETRIES,
    help="Maximum times the researcher can retry a hypothesis if it fails."
)

# --- Main Application ---
st.title("Multi-Agent Research Workflow")
st.subheader("Your AI-Powered Literature Review Assistant")

user_task = st.text_area(
    "Enter your research task:", 
    "I need to find content about AI in healthcare",
    height=100
)

if st.button("Start Research", type="primary"):
    if not user_task:
        st.error("Please enter a research task.")
    else:
        # --- UI Placeholders ---
        st.info("Workflow started... This may take several minutes. See progress below.")
        
        # Placeholders for dynamic content
        progress_container = st.container(border=True)
        st.subheader("Final Report")
        report_container = st.empty()
        
        log_container = progress_container.container(height=350)
        log_container.markdown("### Workflow Log")
        
        status_placeholder = progress_container.empty()
        
        try:
            workflow = build_workflow()
            
            # --- Initial State ---
            initial_state: ResearchState = {
                "user_task": user_task,
                "hypothesis": [],
                "all_papers": [],
                "current_hypothesis_index": 0,
                "retry_counter": 0,
                "max_retries": max_retries,
                "messages": [],
                "current_retrieved_papers": [],
                "validation_results": [],
                "tool_usage_stats": {},
                "final_report": "",
                "workflow_stage": "initial"
            }
            
            # --- Workflow Execution ---
            start_time = time.time()
            final_state = {}

            # Use .stream() to get live updates
            for s in workflow.stream(initial_state):
                # 's' is a dictionary where the key is the node that just ran
                node_name = list(s.keys())[0]
                state_snapshot = s[node_name]
                
                status_placeholder.text(f"Running node: {node_name.upper()}...")
                
                # --- Log Progress to UI ---
                log_container.markdown(f"--- ✅ **{node_name.upper()}** node complete ---")
                
                if node_name == "planner":
                    hyp_count = len(state_snapshot['hypothesis'])
                    log_container.write(f"Generated {hyp_count} hypotheses.")
                    log_container.json([h['text'] for h in state_snapshot['hypothesis']])
                
                elif node_name == "validator":
                    log_container.write("Hypotheses validated by Mellea (if available).")
                    if state_snapshot.get('validation_results'):
                        log_container.json(state_snapshot['validation_results'])

                elif node_name == "researcher":
                    papers_retrieved = len(state_snapshot['current_retrieved_papers'])
                    hyp_idx = state_snapshot['current_hypothesis_index']
                    log_container.write(f"Researched hypothesis {hyp_idx + 1}. Retrieved {papers_retrieved} raw papers.")
                    log_container.json(state_snapshot['tool_usage_stats'])

                elif node_name == "curator":
                    # Curator runs *after* researcher, so index is +1
                    hyp_idx = state_snapshot['current_hypothesis_index']
                    if hyp_idx > 0 and hyp_idx <= len(state_snapshot['hypothesis']):
                        # Look at the hypothesis that was just completed
                        papers_selected = len(state_snapshot['hypothesis'][hyp_idx - 1].get('papers', []))
                        log_container.write(f"Curated results for hypothesis {hyp_idx}. Selected {papers_selected} papers.")
                    else:
                        log_container.write("Curator finished. Moving to next step.")

                elif node_name == "report":
                    log_container.write("Generating final report...")
                
                final_state = state_snapshot # Keep track of the latest state

            # --- Workflow Complete ---
            end_time = time.time()
            status_placeholder.empty()
            st.success(f"Workflow Complete! (Total time: {end_time - start_time:.2f} seconds)")
            
            # Display the final report
            report_container.markdown(final_state.get('final_report', "No report generated."))
            
            # Show tool usage and final papers in sidebar
            st.sidebar.subheader("Workflow Results")
            st.sidebar.metric("Total Papers Selected", len(final_state.get('all_papers', [])))
            
            with st.sidebar.expander("Tool Usage Stats"):
                st.json(final_state.get('tool_usage_stats', {}))
                
            with st.sidebar.expander("Final Paper List (JSON)"):
                st.json(final_state.get('all_papers', []))

        except Exception as e:
            st.error(f"An error occurred during the workflow: {e}")
            import traceback
            st.code(traceback.format_exc())