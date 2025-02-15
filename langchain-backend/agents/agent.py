from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, create_retriever_tool
from langchain_openai import ChatOpenAI

from chroma.chroma import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

@tool
def get_word_length(word: str) -> int:
    """Returns the length of the given word."""
    return len(word)


def get_agent(model="gpt-4o-mini"):
    llm = ChatOpenAI(model=model)

    retriever_tool = create_retriever_tool(
        retriever,
        "document_search",
        "This is the tool which is used to search information in the uploaded documents"
    )

    tools = [get_word_length, retriever_tool]

    llm_with_tools = llm.bind_tools(tools)

    memory_key = "chat_history"
    agent_prompt = ChatPromptTemplate.from_messages([
        (
            "system", """You are an assistant which is capable ONLY of providing the length of the requested word.
             
             If user asks anything different, respond "I only can answer questions about the length of words, I am that dumb for now" """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_tool_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | agent_prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent = agent,
        tools = tools,
        verbose = True
    )

    return agent_executor