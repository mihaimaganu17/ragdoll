import getpass
import os
import bs4

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader


# Initialize the LLM to be used
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

def loader():
    # We load the HTML document using a WebBaseLoader to convert it to a langchain Document and
    # BeatifulSoup to parse the text.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader


def split(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Our document is 42k characters and models struggle to find information in very long input,
    # so we split the document into chunks for embedding and vector storage.
    # This splitter will recursively split the document using commong separactors like new lines
    # until each chunk is the appropriate size.
    text_splitter = RecursiveCharacterTextSplitter(
        # Chunk size in characters
        chunk_size=1000,
        # Chunk overlap in characters. This mitigates loss of information when context is divided
        # between chunks.
        chunk_overlap=200,
        # Track index in original document
        add_start_index=True,
        # Additional punctuation for tricky cases of splitting
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],

    )
    all_splits = text_splitter.split_documents(documents)

    assert len(all_splits) == 63
    return all_splits


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    # Set up and agent for the browser

    # Initialize the model to be used
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    # Load the document we want to search in
    docs = loader().load()
    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")

    # Split the document into smaller chunks, such that it is easier for the context window of the
    # model.
    all_splits = split(docs)

    # Store the splited text chunks into the vector store
    # Embed the contents of each document and store them in the vector store.
    # Given an input query, we can then use vector search to retrieve relevant documents.
    doc_ids = vector_store.add_documents(documents=all_splits)


# Initialize the interface for working with the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# Initialize the vector store for the embeddings
vector_store = InMemoryVectorStore(embeddings)


from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState)

from langchain_core.tools import tool

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query from the vector store."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an assistant message that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or reponse."""
    # Provide available tools for the LLM to call
    llm_with_tools = llm.bind_tools([retrieve])
    # Invoke the LLM with previous messages
    response = llm_with_tools.invoke(state["messages"])
    # MessageState appends messages to state instead of overwriting
    return {"messages": [response]}


from langgraph.prebuilt import ToolNode
# Step 2: Create a tool node to execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generte a response using retrieved content
def generate(state: MessagesState):
    """Generate the answer based on previous messages state"""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break

    # Revert the tools messages back such that we start with the first one
    tool_messages = recent_tool_messages[::-1]

    # Format the docs into a prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        f"{docs_content}"
    )

    # Collect all the conversations message from either the user role, system role or assitant
    # messages which are not tool calls.
    conversation_messages = [
        message for message in state["messages"] if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    from langchain_core.messages import SystemMessage
    # Create an entire new prompt with the system message content and the conversation messages
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # prompt the LLM
    response = llm.invoke(prompt)
    return {"messages": [response]}

from langgraph.graph import END
from langgraph.prebuilt import tools_condition

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    # Standard conditional logic for ReAct-style agents: if the last AI message contains tool calls,
    # route to the tool execution node; otherwise, end the workflow
    tools_condition,
    {END: END, "tools": "tools"},
)

graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
# We compile the graph
graph = graph_builder.compile()

# Save the graph as a .PNG
with open("conv_rag.png", "wb") as g:
    g.write(graph.get_graph().draw_mermaid_png())

input_message = "What is Task Decomposition?"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()