import streamlit as st
from huggingface_hub import InferenceClient
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="AI Resume Builder", layout="wide")

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
Write a strong professional resume summary (3-4 lines).

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

# ------------------------------
# DOCX GENERATOR
# ------------------------------
def generate_docx(profile, summary, experiences):
    doc = Document()

    name = doc.add_paragraph(profile["name"])
    name.runs[0].bold = True
    name.runs[0].font.size = Pt(22)
    name.alignment = WD_ALIGN_PARAGRAPH.CENTER

    contact = doc.add_paragraph(
        f"{profile['email']} | {profile['phone']} | {profile['linkedin']}"
    )
    contact.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Professional Summary", level=2)
    doc.add_paragraph(summary)

    doc.add_heading("Experience", level=2)
    for exp, bullets in experiences:
        p = doc.add_paragraph(f"{exp['title']} – {exp['company']}", style="List Bullet")
        doc.add_paragraph(bullets)

    doc.add_heading("Skills", level=2)
    doc.add_paragraph(", ".join(profile["skills"]))

    file_path = "resume.docx"
    doc.save(file_path)
    return file_path

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

experiences = [
    {
        "title": "AI Intern",
        "company": "Research Lab",
        "description": "Worked on machine learning and NLP projects including classification, clustering, and text analytics"
    }
]

# ------------------------------
# MAIN UI
# ------------------------------
st.title("🤖 AI Resume Builder (HF API)")

generator = ChatbotGenerator()

if st.button("Generate Resume"):
    with st.spinner("Generating resume using Hugging Face API..."):
        summary = generator.resume_summary(profile)
        exp_bullets = [generator.experience_bullets(exp) for exp in experiences]
        cover_letter = generator.cover_letter(profile)

        doc_path = generate_docx(profile, summary, list(zip(experiences, exp_bullets)))

    st.subheader("Resume Summary")
    st.write(summary)

    st.subheader("Cover Letter")
    st.write(cover_letter)

    st.download_button(
        "Download Resume (DOCX)",
        open(doc_path, "rb"),
        file_name="AI_Resume.docx"
    )
