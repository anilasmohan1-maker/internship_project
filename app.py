# app.py

import streamlit as st
from core import load_model, ATSOptimizer, ContentGenerator, DocumentGenerator

st.set_page_config(page_title="AI Resume Builder", layout="centered")
st.title("🤖 AI Resume & Portfolio Builder")

# -----------------------
# LOAD MODEL (CACHED)
# -----------------------
@st.cache_resource
def init_model():
    return load_model()

model, tokenizer = init_model()

# -----------------------
# USER INPUT
# -----------------------
name = st.text_input("Full Name")
email = st.text_input("Email")
location = st.text_input("Location")
skills = st.text_area("Skills (comma separated)")
job_desc = st.text_area("Paste Job Description")

# -----------------------
# GENERATE
# -----------------------
if st.button("Generate Resume"):
    profile = {
        "name": name,
        "email": email,
        "location": location,
        "skills": [s.strip() for s in skills.split(",") if s.strip()]
    }

    ats = ATSOptimizer()
    keywords = ats.extract_keywords(job_desc)

    generator = ContentGenerator(model, tokenizer)

    prompt = f"""
Write a concise ATS-optimized professional summary.

Name: {name}
Skills: {skills}
Keywords: {', '.join(keywords[:8])}

Summary:
"""

    summary = generator.generate_summary(prompt)

    st.subheader("📄 Professional Summary")
    st.write(summary)

    doc_gen = DocumentGenerator()
    html_resume = doc_gen.render_html(profile, summary)
    docx_file = doc_gen.generate_docx(profile, summary)

    st.download_button(
        "⬇️ Download Resume (HTML)",
        html_resume,
        file_name="resume.html"
    )

    st.download_button(
        "⬇️ Download Resume (DOCX)",
        open(docx_file, "rb"),
        file_name="resume.docx"
    )
