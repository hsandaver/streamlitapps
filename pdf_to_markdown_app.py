# Filename: pdf_to_markdown_app.py

import streamlit as st
import pdfplumber
from markdownify import markdownify as md
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="PDF to Markdown Converter",
    page_icon="📄➡️📑",
    layout="centered",
    initial_sidebar_state="auto",
)

# App Title
st.title("📄 PDF to Markdown Converter")

# App Description
st.markdown("""
Convert your PDF documents to Markdown effortlessly. Perfect for preparing content for Large Language Models (LLMs) or for seamless editing and sharing.
""")

# Sidebar for additional options
st.sidebar.header("Options")

# Checkbox to enable OCR
enable_ocr = st.sidebar.checkbox("Enable OCR (For Scanned PDFs)", value=False)

# Text input for Tesseract command path (in case it's not in PATH)
tesseract_cmd_path = st.sidebar.text_input("Tesseract Command Path (optional)")

# Checkbox to show logs in the sidebar
show_logs = st.sidebar.checkbox("Show OCR Logs", value=True)

# Function to extract text using pdfplumber
def extract_text_pdfplumber(pdf_file):
    """Extract text from a PDF using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n# Page {page_num}\n\n" + page_text
    except Exception as e:
        st.error(f"Error while extracting text with pdfplumber: {e}")
        return ""
    return text

# Function to perform OCR on PDF
def extract_text_ocr(pdf_file):
    """Extract text from a scanned PDF using OCR via pytesseract."""
    text = ""
    # Set tesseract_cmd if a custom path is provided
    if tesseract_cmd_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

    with tempfile.TemporaryDirectory() as path:
        try:
            images = convert_from_path(pdf_file, dpi=300, output_folder=path, fmt='png')
        except Exception as e:
            st.error(f"Error converting PDF to images: {e}")
            return ""
        for i, image in enumerate(images):
            if show_logs:
                st.info(f"Performing OCR on page {i+1}...")
            page_text = pytesseract.image_to_string(image)
            text += f"\n\n# Page {i+1}\n\n" + page_text
    return text

# Function to convert text to Markdown
def convert_to_markdown(text):
    """
    Convert raw text to Markdown.
    Using `markdownify` for more sophisticated conversion, 
    but still might need custom tweaking for best results.
    """
    markdown_text = md(text, heading_style="ATX")
    return markdown_text

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing your PDF..."):
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_filename = tmp_file.name

        # Extract text
        if enable_ocr:
            extracted_text = extract_text_ocr(tmp_filename)
        else:
            extracted_text = extract_text_pdfplumber(tmp_filename)

        # Clean up the temporary file
        os.unlink(tmp_filename)

    if extracted_text.strip():
        # Convert to Markdown
        markdown_text = convert_to_markdown(extracted_text)

        # Display the Markdown content
        with st.expander("Preview Markdown"):
            st.markdown(markdown_text)

        # Provide a download button
        st.download_button(
            label="📥 Download Markdown",
            data=markdown_text,
            file_name="output.md",
            mime="text/markdown",
        )
    else:
        st.error("No text extracted from the PDF. Possibly an empty file or an error occurred.")

else:
    st.info("Please upload a PDF file to begin the conversion.")