import os
import time
import lancedb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


def load_documents():
    source, data = prompt_link_or_data()
    if source == "link":
        loader = WebBaseLoader(data)
        documents_loader = None
    else:
        loader = None
        documents_loader = DirectoryLoader(
            "data", glob="./*.pdf", loader_cls=PyPDFLoader
        )

    if loader:
        url_docs = loader.load()
    else:
        url_docs = []

    if documents_loader:
        data_docs = documents_loader.load()
    else:
        data_docs = []

    return url_docs + data_docs


def get_text_from_url(url):

    loader = WebBaseLoader(url)
    url_docs = loader.load()
    data_docs= []
    return url_docs+data_docs

def get_text_from_pdf(documents_loader):
    documents_loader = PyPDFLoader(documents_loader,extract_images=True)
    data_docs = documents_loader.load()
    data_url= []
    return data_docs+data_url


def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
    chunks = text_splitter.split_documents(docs)
    return chunks


def initialize_vector_database(chunks):
    db = lancedb.connect("src/lance_database")
    table = db.create_table(
        "rag_sample",
        data=[
            {
                "vector": OpenAIEmbeddings(model="text-embedding-3-large").embed_query(
                    "Hello World"
                ),
                "text": "Hello World",
                "id": "1",
            }
        ],
        mode="overwrite",
    )
    docsearch = LanceDB.from_documents(
        chunks, OpenAIEmbeddings(model="text-embedding-3-large"), connection=table
    )
    return docsearch.as_retriever(search_kwargs={"k": 3})


def generate_rag_chain(retriever, user_question):
    template = "{query}"
    prompt = ChatPromptTemplate.from_template(template.format(query=user_question))
    rag_chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model="gpt-3.5-turbo-0125")
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
            input("Would you like to enter a link or use existing data? (link/data): ")
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


def main():

    # Load documents
    docs = load_documents()

    # Specify chunk size and overlap
    chunk = get_text_chunks(docs)

    # Initialize Vector Database
    retriever = initialize_vector_database(chunk)

    # Get user question
    user_question = get_user_question()

    # Generate RAG chain
    rag_chain = generate_rag_chain(retriever, user_question)

    # Invoke RAG chain
    response = rag_chain.invoke(user_question)

    # Get complete sentence
    complete_sentence = get_complete_sentence(response)

    print("\nComplete Sentence:")
    print(complete_sentence)


if __name__ == "__main__":
    main()


# def chatpdf(user_question,url):
#     # Process based on user choice
#     if os.path.isfile(url):
#         raw_text = load_documents()
#         text_chunks = get_text_chunks(raw_text)
#         vectorstore = initialize_vector_database(text_chunks)
#         rag_chain = generate_rag_chain(vectorstore,user_question)
#         response = rag_chain.invoke(user_question)
#         complete_sentence = get_complete_sentence(response)
#         return complete_sentence
