from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from transformers import pipeline
import io

app = FastAPI()

# Set your Hugging Face API key
HUGGINGFACE_API_KEY = "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"

# Initialize the Hugging Face pipeline with the API key
hf_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", use_auth_token=HUGGINGFACE_API_KEY)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file: UploadFile):
    pdf_reader = PdfReader(pdf_file.file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Endpoint to upload PDF
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file format. Only PDFs are accepted.")
    text = extract_text_from_pdf(file)
    return {"text": text}

# Model for user query
class Query(BaseModel):
    query: str
    context: str

# Endpoint to get response from LLM
@app.post("/ask/")
async def ask(query: Query):
    response = hf_pipeline(question=query.query, context=query.context)
    answer = response['answer']
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
