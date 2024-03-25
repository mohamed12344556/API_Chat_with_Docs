from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from test1 import *
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader

app = FastAPI()

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


@app.post("/chatpdf/")
async def chatpdf(file: UploadFile = File(...), user_question: str = None):
    # Check if file exists
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Check if the uploaded file is a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(file.file.read())

        # Process the PDF file
        raw_text = get_text_from_pdf("temp.pdf")

        # Process text chunks and conversation chain
        text_chunks = get_text_chunks(raw_text)
        retriever = initialize_vector_database(text_chunks)
        rag_chain = generate_rag_chain(retriever, user_question)
        response = rag_chain.invoke(user_question)
        complete_sentence = get_complete_sentence(response)

        return complete_sentence

    finally:
        # Delete the temporary file
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")


@app.post("/chaturl/")
async def chaturl(user_question: str = None, url: str = None):
    try:
        if user_question is None or url is None:
            raise HTTPException(
                status_code=400, detail="Both user_question and url are required."
            )
        # Get text from the URL
        raw_text = get_text_from_url(url)
        # Process text chunks and conversation chain
        text_chunks = get_text_chunks(raw_text)
        retriever = initialize_vector_database(text_chunks)
        rag_chain = generate_rag_chain(retriever,user_question)
        response = rag_chain.invoke(user_question)
        complete_sentence=get_complete_sentence(response)

        return complete_sentence

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
