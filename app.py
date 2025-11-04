import io
import os
import re
import string
import base64
import concurrent.futures
from functools import lru_cache
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import streamlit as st
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import fitz  # Add this for PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please check your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ------------------------------
# 🔹 Gemini API Call Optimization
# ------------------------------
@st.cache_resource
def get_gemini_response(input_text, resume_text):
    """Send a single request to Gemini AI for all tasks."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompts = {
        "resume_review": """
        You are a senior technical recruiter with expertise in evaluating candidates for roles in Data Science, Full Stack Development, Big Data, DevOps, Cybersecurity, and related domains.

        Your task is to analyze the provided resume in relation to the job description.

        Your response should:
        - Evaluate overall fitness for the role
        - Highlight key strengths relevant to the role
        - Identify weaknesses or missing skills
        - Mention specific areas the candidate should focus on
        - Maintain a professional tone, like real HR feedback

        Format your response with clear bullet points or sections.
        """,

        "match_percentage": """
        Compare the resume and the job description to determine how well they align.

        Your response must include:
        1. A **realistic match percentage** (between 0% and 100%)
        2. A breakdown of:
            - Skills and qualifications that match
            - Relevant experience that matches
            - Key mismatches or gaps (with examples)
        3. Specific suggestions to improve the match

        Format the output like:
        - Match Score: 72%
        - Matching Areas:
        • Python, SQL, Cloud Platforms
        - Missing/Weak Areas:
        • Leadership, Docker, Cybersecurity Basics
        - Suggestions:
        • Add experience with containerization tools like Docker

        Keep the tone friendly yet formal and helpful.
        """,

        "improvement_suggestions": """
        You are an ATS optimization expert with deep knowledge of what hiring systems look for across technical fields such as Data Science, Full Stack Development, DevOps, and Cybersecurity.

        Review the resume in the context of the job description and provide:
        - Specific missing keywords or technical terms
        - Formatting or structural improvements
        - Sections that should be added or revised
        - Skills or certifications that can boost alignment

        Make sure the suggestions are:
        - Practical
        - Action-oriented
        - Tailored to the job description

        Format suggestions in bullet points under clear headings like:
        - 🛠️ Missing Skills
        - 📐 Formatting Improvements
        - 🚀 Boosting Relevance
        """,

        "interview_qa": """
        You are an AI-powered technical recruiter helping candidates prepare for interviews.

        Based on the job description, generate:
        - 5 Technical Questions (with brief model answers)
        - 3 Behavioral Questions (with good sample responses)
        - 2 Situational Questions (with model responses)
        - Bonus: 3 follow-up questions recruiters may ask

        Ensure each question is relevant to the job description. Make the responses:
        - Concise but complete
        - Easy to read
        - Realistic to what's asked in interviews

        Use headings:
        🔧 Technical Questions  
        🤝 Behavioral Questions  
        📊 Situational Questions  
        🔄 Follow-up Questions
        """,
    }

    
    responses = {}

    for key, prompt in prompts.items():
        # Skip keys that are not actual prompt instructions
        if key not in ["resume_review", "match_percentage", "improvement_suggestions", "interview_qa"]:
            continue

        # Build contextual input dynamically
        resume_text = extract_text_from_pdf(uploaded_file)
        context = [
            f"Job Description:\n{input_text}",
            f"Resume Content:\n{resume_text}",
            prompt
        ]

        response = model.generate_content(context)
        responses[key] = response.text

    return responses


# ------------------------------
# 🔹 Parallel PDF Text Extraction
# ------------------------------
@st.cache_resource
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        if len(pdf.pages) > 2:  # Threshold for multi-threading
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = executor.map(lambda page: page.extract_text(), pdf.pages)
        else:
            results = [page.extract_text() for page in pdf.pages]
        text = "\n".join(filter(None, results))
    return text.strip()

# ------------------------------
# 🔹PDF-to-Image Processing
# ------------------------------
@st.cache_resource
def input_pdf_setup(uploaded_file):
    """Convert PDF to an image using PyMuPDF."""
    if uploaded_file is not None:
        # Open the PDF file with PyMuPDF
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        first_page = pdf_document.load_page(0)  # Get the first page

        # Render page to an image (adjust zoom for quality, e.g., 2x = 300 DPI equivalent)
        pix = first_page.get_pixmap(matrix=fitz.Matrix(1, 1))  # 2x zoom ~ 300 DPI

        # Convert to bytes
        img_byte_arr = io.BytesIO(pix.tobytes(output="jpeg"))
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [{"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr).decode()}]
        pdf_document.close()  # Clean up
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

# ------------------------------
# 🔹 Optimized Cosine Similarity with TF-IDF
# ------------------------------
@lru_cache(maxsize=10)
def get_vectorizer():
    return TfidfVectorizer()

# ------------------------------
# 🔹 Matching Score
# ------------------------------
@st.cache_resource
def calculate_match_score(resume_text, job_description):
    """Calculate match score using cached TF-IDF vectorizer."""
    if not resume_text or not job_description:
        return 0  

    vectorizer = get_vectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description]).toarray()
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    
    return round(cosine_sim * 100, 2)

def get_short_title(text, max_chars=80):
    if len(text) <= max_chars:
        return text
    cutoff = text[:max_chars].rfind(' ')
    return text[:cutoff] if cutoff != -1 else text[:max_chars]

# Function to generate PDF report
def generate_pdf_report(responses, match_score, job_title="Job Description Analysis"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = styles["Title"]
    heading_style = styles["Heading1"]
    body_style = styles["BodyText"]
    body_style.fontSize = 10
    body_style.leading = 12  # Line spacing

    # Title with dynamic job title
    story.append(Paragraph(f"A JobFitAI report for - {job_title}", title_style))
    story.append(Spacer(1, 18))

    # Helper to clean markdown and split text
    def clean_and_split(text):
        # Remove markdown symbols
        text = re.sub(r'[#*`]+', '', text)  # Strip #, *, `
        # Split into paragraphs on double newlines or long lines
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    # Resume Review
    story.append(Paragraph("Resume Review", heading_style))
    story.append(Spacer(1, 6))
    review_paragraphs = clean_and_split(responses["resume_review"])
    for para in review_paragraphs:
        story.append(Paragraph(para, body_style))
        story.append(Spacer(1, 6))

    # Match Percentage
    story.append(Paragraph("Match Percentage", heading_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Resume Match Score: {match_score}%", body_style))
    story.append(Spacer(1, 12))

    # Improvement Suggestions
    story.append(Paragraph("Improvement Suggestions", heading_style))
    story.append(Spacer(1, 6))
    suggestions_paragraphs = clean_and_split(responses["improvement_suggestions"])
    for para in suggestions_paragraphs:
        story.append(Paragraph(para, body_style))
        story.append(Spacer(1, 6))

    # Interview Q&A
    story.append(Paragraph("Interview Questions & Answers", heading_style))
    story.append(Spacer(1, 6))
    qa_paragraphs = clean_and_split(responses["interview_qa"])
    for para in qa_paragraphs:
        story.append(Paragraph(para, body_style))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

def preprocess_text(text):
    """Preprocess text by removing noise and standardizing."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# ------------------------------
# 🔹 Streamlit UI
# ------------------------------
st.set_page_config(page_title="JobFitAI")
st.header("📄 JobFitAI - Smart Resume Analyzer & Optimizer")
job_title_input = st.text_input("Enter Job Title (e.g., Software Engineer):", "Resume Analysis")
# User Inputs
input_text = st.text_area("📝 Enter Job Description:", key="input")
uploaded_file = st.file_uploader("📂 Upload Your Resume (PDF)", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF Uploaded Successfully!")

# Action Buttons
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    submit1 = st.button("📊 Resume Analysis")
with col2:
    submit2 = st.button("📈 Match Percentage")
with col3:
    submit3 = st.button("📌 Improvement Suggestions")
with col4:
    submit4 = st.button("🎤 Interview Q&A")
with col5:
    submit5 = st.button("📥 Export Report")

# ------------------------------
# 🔹 API Processing & Response
# ------------------------------
if uploaded_file and input_text:
    # Precompute data only when a button is pressed
    if submit1 or submit2 or submit3 or submit4 or submit5:
        with st.spinner("🔍 Processing your request... Please wait!"):
            pdf_content = input_pdf_setup(uploaded_file)
            responses = get_gemini_response(input_text, pdf_content)
            resume_text = extract_text_from_pdf(uploaded_file)
            match_score = calculate_match_score(resume_text, input_text)

            # Resume Review
            if submit1:
                st.subheader("📝 Resume Review")
                st.write(responses["resume_review"])
                # st.success("✅ Done!")

        # Match Percentage
            elif submit2:                
                st.subheader("📊 Match Score")
                st.write(f"🔍 Resume Match Score: **{match_score}%**")
                st.write(responses["match_percentage"])
                # st.success("✅ Done!")

                # Visualization
                df = pd.DataFrame([{"Category": "Resume Match", "Score": match_score}])
                fig = px.bar(
                    df,
                    x="Category",
                    y="Score",
                    text=[f"{match_score}%"],  # Custom text on bar
                    title="Resume vs Job Description Match Analysis",
                    color="Score",
                    color_continuous_scale="RdYlGn",  # Red-Yellow-Green for intuitive scoring
                    range_y=[0, 100],  # Fix y-axis to 0-100% for consistency
                )

                # Customize layout
                fig.update_traces(
                    textposition="auto",  # Position text nicely
                    width=0.4,  # Narrower bars for aesthetics
                )
                fig.update_layout(
                    yaxis_title="Match Percentage (%)",
                    xaxis_title="",
                    showlegend=False,  # Hide legend since it’s just one bar
                    plot_bgcolor="white",
                    # Add a dashed line for a "good match" threshold
                    shapes=[
                        dict(
                            type="line",
                            y0=70, y1=70, x0=-0.5, x1=0.5,  # Adjusted for single bar
                            line=dict(color="gray", width=2, dash="dash"),
                        )
                    ],
                    annotations=[
                        dict(
                            x=0, y=70, text="Good Match Threshold (70%)",
                            showarrow=False, yshift=10, font=dict(color="gray")
                        )
                    ]
                )

                st.plotly_chart(fig, use_container_width=True)

                # Optional: Add a simple interpretation
                if match_score >= 70:
                    st.success("✅ Your resume aligns well with the job description!")
                elif match_score >= 50:
                    st.warning("⚠ Your resume has moderate alignment—check suggestions.")
                else:
                    st.error("❌ Low match—consider revising your resume.")

                    # Resume Improvement Suggestions
            elif submit3:
                st.subheader("📌 Resume Improvement Suggestions")
                st.write(responses["improvement_suggestions"])
                # st.success("✅ Done!")

                    # Interview Questions & Answers
            elif submit4:
                st.subheader("🎤 Interview Q&A")
                st.write(responses["interview_qa"])
                # st.success("✅ Done!")

            elif submit5:
                st.subheader("📥 Export Report")
                pdf_buffer = generate_pdf_report(responses, match_score, job_title=job_title_input)
                if pdf_buffer.getbuffer().nbytes > 0:  # Verify buffer isn’t empty
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer,
                        file_name="resume_analysis_report.pdf",
                        mime="application/pdf",
                        key="download_report"  # Unique key to avoid conflicts
                    )
                    st.success("✅ Report generated! Click above to download.")
                else:
                    st.error("❌ Failed to generate PDF - buffer is empty.")
else:
    st.warning("⚠ Please upload a resume and provide a job description.")
