import getpass
import os

from langchain.chat_models import init_chat_model

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")

if __name__ == "__main__":
    main()
