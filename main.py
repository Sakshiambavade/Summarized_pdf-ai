import streamlit as st
import pdfplumber
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from llama_index.core.prompts import PromptTemplate
import os

# Load the environment variables from .env file
load_dotenv()

# Fetch the API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM function
def initialize_llm(model_type):
    return Groq(model=model_type, api_key=api_key)

# Summarization function using `llm.complete()`
def summarize_text(llm, text, summary_type):
    prompts = {
        "Long Summary": "Summarize the following text in detail:\n{text}",
        "Short Summary": "Summarize the following text in 100 words:\n{text}",
        "Creative Summary": "Provide a creative summary of the following text:\n{text}",
        "Bullet Point Summary": "Summarize the following text in 3 bullet points:\n{text}"
    }

    text = text[:5000]  # Limit input to first 5000 characters

    # Format the prompt
    formatted_prompt = prompts[summary_type].format(text=text)

    # Get response from LLM using `complete()`
    response = llm.complete(formatted_prompt)

    # Debugging: Print full response
    print("LLM Response:", response)

    return response.text  # Extract only the generated text

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n\n"
    return text.strip()

# Streamlit App
st.title("📄 AI-Powered PDF Summarizer 🤖")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Extract text from the uploaded PDF
if uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file)
else:
    extracted_text = ""

# Collapsible extracted text
with st.expander("🔍 View Extracted Text"):
    st.text_area("Extracted Text", value=extracted_text, height=300, disabled=True)

# Summary Type Selection
summary_type = st.selectbox(
    "Select Summary Type",
    ("Long Summary", "Short Summary", "Creative Summary", "Bullet Point Summary")
)

# Model Type Selection
model_type = st.selectbox(
    "Select Model Type",
    ("qwen-2.5-32b", "llama3-70b-8192", "deepseek-r1-distill-qwen-32b")
)

# Initialize the selected model
llm = initialize_llm(model_type)

# Button to generate summary
if st.button("Generate Summary"):
    if extracted_text:
        with st.spinner("Generating summary... ⏳"):
            summary = summarize_text(llm, extracted_text, summary_type)
            st.subheader(f"📜 {summary_type} using {model_type}")
            st.write(summary)
    else:
        st.warning("⚠️ Please upload a PDF first.")

# Footer
st.markdown("---")
st.markdown("🚀 Made by **Sakshi Ambavade**")
