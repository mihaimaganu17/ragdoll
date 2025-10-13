import uuid

from typing import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


class State(TypedDict):
    text_1: str
    text_2: str


def human_node_1(state: State):
    value = interrupt({"text_to_revise": state["text_1"]})
    return {"text_1": value}


def human_node_2(state: State):
    value = interrupt({"text_to_revise": state["text_2"]})
    return {"text_2": value}


graph_builder = StateGraph(State)
graph_builder.add_node("human_node_1", human_node_1)
graph_builder.add_node("human_node_2", human_node_2)

# Add parallel edges from START to the 2 interrupt nodes
graph_builder.add_edge(START, "human_node_1")
graph_builder.add_edge(START, "human_node_2")

memory_checkpoint = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory_checkpoint)

thread_id = str(uuid.uuid4())
config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
result = graph.invoke(
    {"text_1": "original 1", "text_2": "original text 2"}, config=config
)

resume_map = {}

# Resume with mapping of interrupt IDs to values
for i in graph.get_state(config).interrupts:
    resume_map[i.id] = f"edited text for {i.value['text_to_revise']}"


print(graph.invoke(Command(resume=resume_map), config=config))