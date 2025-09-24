import getpass
import os
import bs4

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader


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
    # Initialize the interface for working with the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # Initialize the vector store for the embeddings
    vector_store = InMemoryVectorStore(embeddings)

    # Load the document we want to search in
    docs = loader().load()
    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")

    split(docs)
    

if __name__ == "__main__":
    main()
