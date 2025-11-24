# Code with a agent and custom tool implementation

import subprocess
import sys
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, END, START, StateGraph

llama = ChatOllama(
    model="llama3.1:8b",
    temperature=0.7,
)


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

@tool
def multiply(a: float, b: float)-> float:
    ''' Multiply function tool 
        Parameters;
            a: float number desired for multiplication
            b: second float number desired for multiplication

        returns 
            result of the multiplication of the parameters 
    '''
    return a * b 

tools = [search_arXiv, multiply]

tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llama.bind_tools(tools)

def llm_call(state: MessagesState ):

    return {
        'messages': [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful researcher for papers based on a asked topic use only your tools to get papers information, you are not allowed to generate fake data of internal inforamtion unless it is from your tools source"
                    )
                ]
                + state["messages"]
            )
        ]
    }

# tool node 
def tool_node(state: dict):

    result = []
    for tool_call in state["messages"][-1].tool_calls: 

        tool = tools_by_name[tool_call['name']]
        observation = tool.invoke(tool_call['args'])
        result.append(ToolMessage(content=observation, tool_call_id= tool_call['id']))

    return {"messages": result}

# conditional node 
def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:

    message = state['messages'][-1]

    # if the model wanna make a tool call 
    if message.tool_calls:
        return "tool_node"

    return END 


agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue, 
    ["tool_node", END]
)

agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

# agent.get_graph(xray=True).draw_mermaid_png()

messages = [HumanMessage(content='Find 3 papers about food and technology')]
messages = agent.invoke({"messages": messages})

for m in messages["messages"]:
    m.pretty_print()


