from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from io import BytesIO
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from test1 import *
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from pydantic import BaseModel
import io
import time
import shutil


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserQuestion(BaseModel):
    question: str


memory = ConversationBufferMemory()


@app.get("/")
async def root():
    return {"message": "Hello, world!"}


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    start_time = time.time()  # Start time
    contents = await file.read()
    pdf_bytes = io.BytesIO(contents)
    if not contents:
        raise HTTPException(
            status_code=400, detail="Failed to load documents from the file."
        )

    # Ensure the "data" directory exists or create it if not
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save the uploaded file to "data" directory
    file_path = os.path.join(data_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    response_time = time.time() - start_time
    return {
        "message": "PDF uploaded successfully",
        "pdf_bytes": pdf_bytes,
        "response_time": response_time,
    }


@app.post("/chatpdf/")
async def process_user_question(user_question: UserQuestion):
    start_time = time.time()
    try:
        # Load documents from the file
        docs = load_documents_from_file(upload_pdf())

        if not docs:
            raise HTTPException(
                status_code=400, detail="No documents loaded from file."
            )

        # Specify chunk size and overlap
        chunk = get_text_chunks(docs)

        # Initialize Vector Database
        retriever = initialize_vector_database(chunk)

        # Generate RAG chain
        rag_chain = generate_rag_chain(retriever, user_question.question, memory)

        # Invoke RAG chain
        response = ""
        for token in rag_chain.stream(user_question.question):
            response += token

        # Get complete sentence
        complete_sentence = get_complete_sentence(response)

        # Save the interaction in memory
        memory.chat_memory.add_message({"role": "user", "content": user_question})
        memory.chat_memory.add_message(
            {"role": "assistant", "content": complete_sentence}
        )

        response_time = time.time() - start_time
        return {"response": complete_sentence, "response_time": response_time}
    except Exception as e:
        response_time = time.time() - start_time
        raise HTTPException(
            status_code=500, detail={"error": str(e), "response_time": response_time}
        )


@app.post("/chaturl/")
async def chaturl(user_question: UserQuestion, url: str):
    start_time = time.time()
    try:
        if not user_question or not url:
            raise HTTPException(
                status_code=400, detail="Both user_question and url are required."
            )

        # Get text from the URL
        docs = load_documents_from_url(url)
        if not docs:
            raise HTTPException(
                status_code=400, detail="Failed to load documents from the URL."
            )

        # Specify chunk size and overlap
        text_chunks = get_text_chunks(docs)

        # Initialize Vector Database
        retriever = initialize_vector_database(text_chunks)

        # Generate RAG chain
        rag_chain = generate_rag_chain(retriever, user_question.question, memory)

        # Invoke RAG chain
        response = ""
        for token in rag_chain.stream(user_question.question):
            response += token

        # Get complete sentence
        complete_sentence = get_complete_sentence(response)
        # Save the interaction in memory
        memory.chat_memory.add_message({"role": "user", "content": user_question})
        memory.chat_memory.add_message(
            {"role": "assistant", "content": complete_sentence}
        )

        response_time = time.time() - start_time
        return {"response": complete_sentence, "response_time": response_time}
    except Exception as e:
        response_time = time.time() - start_time
        raise HTTPException(
            status_code=500, detail={"error": str(e), "response_time": response_time}
        )


# @app.get("/chathistory/")
# async def get_chat_history():
#     chat_history = [
#         {"role": message["role"], "content": message["content"]}
#         for message in memory.chat_memory.messages
#     ]
#     return {"chat_history": chat_history}
