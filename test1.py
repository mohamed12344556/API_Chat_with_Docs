import os
from dotenv import load_dotenv
import lancedb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def load_documents():
    source, data = prompt_link_or_data()
    documents = []
    try:
        if source == "link":
            loader = WebBaseLoader(data)
            documents = loader.load()
        else:
            documents_loader = DirectoryLoader(
                "data", glob="./*.pdf", loader_cls=PyPDFLoader
            )
            documents = documents_loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
    return documents


def load_documents_from_file(documents_loader):
    documents_loader = DirectoryLoader("data", glob="./*.pdf", loader_cls=PyPDFLoader)
    documents_docs = documents_loader.load()
    return documents_docs


def load_documents_from_url(url: str):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading documents from URL: {e}")
        return []


def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000000, chunk_overlap=10000
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def initialize_vector_database(chunks):
    db = lancedb.connect("src/lance_database")
    table = db.create_table(
        "rag_sample",
        data=[
            {
                "vector": GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                ).embed_query("Hello World"),
                "text": "Hello World",
                "id": "1",
            }
        ],
        mode="overwrite",
    )
    docsearch = LanceDB.from_documents(
        chunks,
        GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        connection=table,
    )
    return docsearch.as_retriever(search_kwargs={"k": 3})


def generate_rag_chain(retriever, user_question, memory):
    history = "\n".join(
        [
            f"{message['role']}: {message['content']}"
            for message in memory.chat_memory.messages
        ]
    )
    template = "{history}\n\nUser: {query}\nAssistant:"
    prompt = ChatPromptTemplate.from_template(
        template.format(history=history, query=user_question)
    )

    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | ChatGoogleGenerativeAI(model="gemini-pro")
        | StrOutputParser()
    )
    return rag_chain


def get_complete_sentence(response):
    last_period_index = response.rfind(".")
    if last_period_index != -1:
        return response[: last_period_index + 1]
    else:
        return response


def get_user_question():
    return input("Please enter your question: ")


def prompt_link_or_data():
    while True:
        choice = (
            input(
                "Would you like to enter a link or use existing data? (link / data): "
            )
            .strip()
            .lower()
        )
        if choice == "link":
            link = input("Please enter the link: ").strip()
            return "link", link
        elif choice == "data":
            return "data", None
        else:
            print("Invalid choice. Please enter 'link' or 'data'.")


def print_chat_history(memory):
    print("\nChat History:")
    for message in memory.chat_memory.messages:
        print(f"{message['role']}: {message['content']}")
    print("\n")


def main():
    memory = ConversationBufferMemory()

    # Load documents
    docs = load_documents()
    if not docs:
        print("No documents loaded. Exiting.")
        return

    # Specify chunk size and overlap
    chunk = get_text_chunks(docs)

    # Initialize Vector Database
    retriever = initialize_vector_database(chunk)

    while True:
        # Get user question
        user_question = get_user_question()

        # Generate RAG chain
        rag_chain = generate_rag_chain(retriever, user_question, memory)

        # Invoke RAG chain and handle streaming output
        response = ""
        try:
            for token in rag_chain.stream(user_question):
                print(token, end="", flush=True)
                response += token

            print("\n")  # Print a new line after the streamed response

            # Get complete sentence
            complete_sentence = get_complete_sentence(response)

            # Save the interaction in memory
            memory.chat_memory.add_message({"role": "user", "content": user_question})
            memory.chat_memory.add_message(
                {"role": "assistant", "content": complete_sentence}
            )
            # Print chat history
            # print_chat_history(memory)
        except Exception as e:
            print(f"Error during RAG chain execution: {e}")


# if __name__ == "__main__":
#     main()
