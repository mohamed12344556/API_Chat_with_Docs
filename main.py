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

app = FastAPI()


# Define request models
class UserQuestion(BaseModel):
    question: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello, world!"}


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    pdf_bytes = io.BytesIO(contents)
    return pdf_bytes


# @app.post("/chatpdf/")
# async def chatpdf(file: UploadFile = File(...), user_question: str = None):
#     # Check if file exists
#     if not file:
#         raise HTTPException(status_code=400, detail="No file uploaded")

#     # Check if the uploaded file is a PDF
#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are allowed")

#     try:
#         # Save the uploaded file temporarily
#         with open("temp.pdf", "wb") as f:
#             raw_text = get_text_from_pdf(f.write(file.file.read()))

#         # Process text chunks and conversation chain
#         text_chunks = get_text_chunks(raw_text)
#         retriever = initialize_vector_database(text_chunks)
#         rag_chain = generate_rag_chain(retriever, user_question)
#         response = rag_chain.invoke(user_question)
#         complete_sentence = get_complete_sentence(response)

#         return complete_sentence

#     finally:
#         # Delete the temporary file
#         if os.path.exists("temp.pdf"):
#             os.remove("temp.pdf")


@app.post("/chatpdf/")
async def process_user_question(user_question: UserQuestion):
    # Load documents
    docs = get_text_from_pdf(upload_pdf())

    # Specify chunk size and overlap
    chunk = get_text_chunks(docs)

    # Initialize Vector Database
    retriever = initialize_vector_database(chunk)

    # Generate RAG chain
    rag_chain = generate_rag_chain(retriever, user_question.question)

    # Invoke RAG chain
    response = rag_chain.invoke(user_question.question)

    # Get complete sentence
    complete_sentence = get_complete_sentence(response)

    return {"response": complete_sentence}


@app.post("/chaturl/")
async def chaturl(user_question: UserQuestion, url: str = None):
    try:
        if user_question is None or url is None:
            raise HTTPException(
                status_code=400, detail="Both user_question and url are required."
            )
        # Get text from the URL
        url_text = get_text_from_url(url)

        # Specify chunk size and overlap
        text_chunks = get_text_chunks(url_text)

        # Initialize Vector Database
        retriever = initialize_vector_database(text_chunks)

        # Generate RAG chain
        rag_chain = generate_rag_chain(retriever, user_question.question)

        # Invoke RAG chain
        response = rag_chain.invoke(user_question.question)
        
        # Get complete sentence
        complete_sentence = get_complete_sentence(response)

        return {"response": complete_sentence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
