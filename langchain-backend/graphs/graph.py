from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from langgraph.graph import MessagesState, END, StateGraph
from langgraph.types import Command

memory = MemorySaver()

members = ["sum-expert", "get-word-length-expert"]

# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
    " If worker is not able to complete the task, go to the next worker available worker."
    " You are a system which can help with summing numbers and getting the length of words."
    " If user asks about anything else, immediately skip all workers and respond with what the system can help with."
)

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["sum-expert", "get-word-length-expert", "FINISH"]

llm = ChatOpenAI(model="gpt-4o-mini")

class State(MessagesState):
    next: str

def supervisor_node(state: State) -> Command[Literal["sum-expert", "get-word-length-expert", "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

# SUM EXPERT AGENT
@tool
def sum_tool(numbers: list[int]) -> int:
    """Returns the sum of the given numbers."""
    return sum(numbers)

sum_expert_agent = create_react_agent(
    llm,
    tools=[sum_tool],
    prompt="""
    Use this worker only for if user asks for summing numbers. If asked about anything else, skip this worker.
    """
)

def sum_expert_node(state: State) -> Command[Literal["supervisor"]]:
    result = sum_expert_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="sum-expert")
            ]
        },
        goto="supervisor",
    )

# GET WORD LENGTH EXPERT AGENT

@tool
def get_word_length(word: str) -> int:
    """Returns the length of the given word."""
    return len(word)

get_word_length_agent = create_react_agent(
    llm,
    tools=[get_word_length],
    prompt="""
    Use this worker only for if user asks about word length. If asked about anything else, skip this worker.
    """
)

def get_word_length_node(state: State) -> Command[Literal["supervisor"]]:
    result = get_word_length_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="get-word-length-expert")
            ]
        },
        goto="supervisor",
    )

builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("sum-expert", sum_expert_node)
builder.add_node("get-word-length-expert", get_word_length_node)
graph = builder.compile(checkpointer=memory)



