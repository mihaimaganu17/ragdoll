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
# The Command primitive resumes execution when it is supplied via invoke.
# At this point the graph resumes execution from the beginning of the node containing the
# `interrupt` call, but this time the interrupt function will return the value provided in 
# Command(resume=value) rather than pausing again.
print(graph.invoke(Command(resume="Edited_text"), config=config))