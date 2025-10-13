from typing import TypedDict
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

class State(TypedDict):
    text: str


def human_node(state: State):
    value = interrupt (
        {
            "text_to_revise": state["text"]
        }
    )
    return {
        "text": value,
    }


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("human_node", human_node)
graph_builder.add_edge(START, "human_node")

memory_checkpoint = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory_checkpoint)
# Create a config to run the graph's thread
config = {"configurable": {"thread_id": uuid.uuid4()}}
# Run the graph until the interrupt is hit.
result = graph.invoke({"text": "You either create or die"}, config=config)

print(result['__interrupt__'])
print(graph.invoke(Command(resume="Edited_text"), config=config))