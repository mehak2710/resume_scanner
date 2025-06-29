import streamlit as st
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Extract Text from PDF ---
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# --- Extract Text from DOCX ---
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# --- Calculate Similarity ---
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

# --- Streamlit UI ---
st.title("📄 Resume Scanner")
st.write("Upload your resume in PDF, DOCX, or TXT format to compare with the job description.")

uploaded_file = st.file_uploader(
    "Upload your resume (.pdf, .docx, .txt)", 
    type=["pdf", "docx", "txt"]
)

# Load job description
with open("job_description.txt", "r", encoding="utf-8") as f:
    job_desc = f.read().lower()

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == "pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        resume_text = extract_text_from_docx(uploaded_file)
    elif file_type == "txt":
        resume_text = uploaded_file.read().decode("utf-8")
    else:
        st.error("❌ Unsupported file format.")
        st.stop()

    if resume_text.strip():
        score = calculate_similarity(resume_text.lower(), job_desc)
        st.success(f"✅ Match Score: {score:.2f}%")
    else:
        st.warning("⚠️ Couldn't extract readable text from the file.")
