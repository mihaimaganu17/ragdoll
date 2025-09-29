import getpass
import os
import bs4

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict


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


def base_prompt():
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt", api_url="https://api.smith.langchain.com")

    example_messages = prompt.invoke(
        {"context": "(context goes here)", "question": "(question goes here)"}
    ).to_messages()

    assert len(example_messages) == 1
    return prompt


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

    # Initialize the base prompt
    prompt = base_prompt()

    # We use LangGraph to to tie together the retrievel and generation steps for the RAG into a
    # single application.
    # To use LangGraph, we need to define 3 things:
    # 1. The state of our application
    # 2. The nodes of our application (application steps)
    # 3. The "control flow" of our application (ordering the steps)

    

# Initialize the interface for working with the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# Initialize the vector store for the embeddings
vector_store = InMemoryVectorStore(embeddings)

# State controls what data is input to the application, trasferred between steps and output by
# the application. For a simple RAG application, we can just keep track of the input question,
# retrieved context and generated answer.
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Retrieve the similar records with the question from the passed state
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


# The generation step formats the retrieved context and original question into a prompt for the chat
# model.
def generate(state: State):
    # Join all the documents in the state's context
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = base_prompt()
    # Replace the question and the retrieved context in the prompt
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    # Call the LLM to get the response
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    response = llm.invoke(messages)
    return { "answer": response.content }


from langgraph.graph import START, StateGraph

# We create a graph with the retrieve and generation steps into a single sequence and then we
# compile it. First we add the 2 Nodes as application steps in a sequence
# Is 'StateGraph' an automaton?
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# We add the start and connect it to the first node "retrieve"
graph_builder.add_edge(START, "retrieve")
# We compile the graph
graph = graph_builder.compile()

# Save the graph as a .PNG
with open("graph.png", "wb") as g:
    g.write(graph.get_graph().draw_mermaid_png())

if __name__ == "__main__":
    main()

    # Invoke
    result = graph.invoke({"question": "What is Task Decomposition"})
    print(f"Context: {result["context"]}\n\n")
    print(f"Answer: {result['answer']}")

    # Async invocations:
    # result = await graph.ainvoke(...)

    # Stream steps
    for step in graph.stream({"question": "What is Task Decomposition?"}, stream_mode="updates"):
        print(f"{step}\n\n-------\n")

    # Async streaming
    # async for step in graph.astream(...):

    # Stream tokens
    for message, metadata in graph.stream({"question": "What is Task Decomposition?"}, stream_mode="messages"):
        print(message.content, end="|")