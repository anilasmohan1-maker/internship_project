import streamlit as st
from huggingface_hub import InferenceClient
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path
import zipfile
import json
from datetime import datetime

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="AI Resume & Portfolio Builder", layout="wide")

# ------------------------------
# HF INFERENCE CLIENT
# ------------------------------
HF_API_KEY = st.secrets.get("HF_API_KEY")
if not HF_API_KEY:
    st.error("HF_API_KEY not found in Streamlit secrets.")
    st.stop()

client = InferenceClient(
    model="togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    token=HF_API_KEY
)

# ------------------------------
# CHATBOT GENERATOR
# ------------------------------
class ChatbotGenerator:
    def chat(self, system_prompt, user_prompt, max_tokens=300):
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()

    def resume_summary(self, profile):
        return self.chat(
            system_prompt="You are a professional resume writer.",
            user_prompt=f"""
Write a realistic, ATS-optimized professional summary (3-4 lines).

Candidate Name: {profile['name']}
Target Role: {profile['role']}
Skills: {', '.join(profile['skills'])}
Experience Level: {profile['experience_level']}
"""
        )

    def experience_bullets(self, exp):
        return self.chat(
            system_prompt="You write impactful resume bullet points.",
            user_prompt=f"""
Create 3-4 resume bullet points for the following role:

Role: {exp['title']}
Company: {exp['company']}
Description: {exp['description']}
"""
        )

    def cover_letter(self, profile):
        return self.chat(
            system_prompt="You are a professional HR cover letter writer.",
            user_prompt=f"""
Write a concise professional cover letter (3 short paragraphs).

Name: {profile['name']}
Role: {profile['role']}
Company: {profile['company']}
Skills: {', '.join(profile['skills'])}
"""
        )

    def portfolio_html(self, profile, experiences, summary):
        bullets_html = ""
        for exp, bullet_text in experiences:
            bullets_html += f"<h4>{exp['title']} - {exp['company']}</h4><p>{bullet_text}</p>"
        html_content = f"""
        <html>
        <head><title>{profile['name']} Portfolio</title></head>
        <body>
        <h1>{profile['name']}</h1>
        <p>{profile['email']} | {profile['phone']} | {profile['linkedin']}</p>
        <h2>Professional Summary</h2>
        <p>{summary}</p>
        <h2>Experience</h2>
        {bullets_html}
        <h2>Skills</h2>
        <p>{', '.join(profile['skills'])}</p>
        </body>
        </html>
        """
        return html_content

# ------------------------------
# DOCX GENERATOR
# ------------------------------
def generate_docx(filename, profile, summary, experiences, cover_letter_text=None):
    doc = Document()

    # Header
    name = doc.add_paragraph(profile["name"])
    name.runs[0].bold = True
    name.runs[0].font.size = Pt(22)
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER

    contact = doc.add_paragraph(
        f"{profile['email']} | {profile['phone']} | {profile['linkedin']}"
    )
    contact.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Resume summary
    doc.add_heading("Professional Summary", level=2)
    doc.add_paragraph(summary)

    # Experience
    doc.add_heading("Experience", level=2)
    for exp, bullets in experiences:
        doc.add_paragraph(f"{exp['title']} – {exp['company']}", style="List Bullet")
        doc.add_paragraph(bullets)

    # Skills
    doc.add_heading("Skills", level=2)
    doc.add_paragraph(", ".join(profile["skills"]))

    # Cover letter if needed
    if cover_letter_text:
        doc.add_page_break()
        doc.add_heading("Cover Letter", level=1)
        doc.add_paragraph(cover_letter_text)

    doc.save(filename)
    return filename

# ------------------------------
# ZIP GENERATOR
# ------------------------------
def create_zip(files, zip_name="portfolio.zip"):
    with zipfile.ZipFile(zip_name, "w") as z:
        for fpath, arcname in files:
            z.write(fpath, arcname)
    return zip_name

# ------------------------------
# SIDEBAR INPUT
# ------------------------------
st.sidebar.title("Candidate Details")

profile = {
    "name": st.sidebar.text_input("Full Name", "Anila R"),
    "email": st.sidebar.text_input("Email", "anila@email.com"),
    "phone": st.sidebar.text_input("Phone", "+91 90000 00000"),
    "linkedin": st.sidebar.text_input("LinkedIn", "linkedin.com/in/anila"),
    "role": st.sidebar.text_input("Target Role", "Machine Learning Engineer"),
    "company": st.sidebar.text_input("Target Company", "Tech Company"),
    "experience_level": st.sidebar.selectbox(
        "Experience Level", ["Fresher", "1-3 Years", "3-5 Years"]
    ),
    "skills": st.sidebar.multiselect(
        "Skills",
        ["Python", "Machine Learning", "Deep Learning", "NLP", "SQL", "TensorFlow"],
        default=["Python", "Machine Learning", "NLP"]
    )
}

num_exp = st.sidebar.slider("Number of Experiences", 1, 3, 1)
experiences = []
for i in range(num_exp):
    st.sidebar.markdown(f"### Experience {i+1}")
    exp_title = st.sidebar.text_input(f"Job Title {i+1}", "AI Intern")
    exp_company = st.sidebar.text_input(f"Company {i+1}", "Research Lab")
    exp_desc = st.sidebar.text_area(f"Description {i+1}", "Worked on ML/NLP projects.")
    experiences.append({"title": exp_title, "company": exp_company, "description": exp_desc})

# ------------------------------
# MAIN UI
# ------------------------------
st.title("🤖 AI Resume & Portfolio Builder (HF API)")

generator = ChatbotGenerator()

if st.button("Generate Resume, Portfolio & Cover Letter"):
    with st.spinner("Generating content via Hugging Face API..."):
        # Resume summary
        summary = generator.resume_summary(profile)
        # Experience bullets
        exp_bullets = [generator.experience_bullets(exp) for exp in experiences]
        # Cover letter
        cover_letter_text = generator.cover_letter(profile)
        # Portfolio HTML
        portfolio_html = generator.portfolio_html(profile, list(zip(experiences, exp_bullets)), summary)

        # Generate DOCX files
        resume_docx = generate_docx("resume.docx", profile, summary, list(zip(experiences, exp_bullets)))
        cover_docx = generate_docx("cover_letter.docx", profile, summary, [], cover_letter_text)
        portfolio_docx = generate_docx("portfolio.docx", profile, summary, list(zip(experiences, exp_bullets)))

        # Create ZIP
        zip_file = create_zip([
            ("resume.docx", "resume.docx"),
            ("cover_letter.docx", "cover_letter.docx"),
            ("portfolio.docx", "portfolio.docx"),
            ("portfolio.html", "portfolio.html")
        ])

    # ------------------------------
    # DISPLAY
    # ------------------------------
    st.subheader("Resume Summary")
    st.write(summary)

    st.subheader("Cover Letter")
    st.write(cover_letter_text)

    st.subheader("Portfolio Preview")
    st.components.v1.html(portfolio_html, height=500, scrolling=True)

    st.download_button("Download Resume (DOCX)", open(resume_docx, "rb"), file_name="resume.docx")
    st.download_button("Download Cover Letter (DOCX)", open(cover_docx, "rb"), file_name="cover_letter.docx")
    st.download_button("Download Portfolio (DOCX)", open(portfolio_docx, "rb"), file_name="portfolio.docx")
    st.download_button("Download All Files (ZIP)", open(zip_file, "rb"), file_name="portfolio.zip")
