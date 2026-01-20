import streamlit as st
import json
import zipfile
from datetime import datetime
from pathlib import Path
from collections import Counter

import nltk
import torch
from jinja2 import Environment, BaseLoader
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from transformers import AutoTokenizer, AutoModelForCausalLM

from samples import SAMPLE_PROFILES

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Resume & Portfolio Builder",
    layout="wide"
)

# -------------------------------------------------
# NLTK SETUP (CACHED)
# -------------------------------------------------
@st.cache_resource
def setup_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")

setup_nltk()

# -------------------------------------------------
# MODEL LOADING (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------------------------
# ATS OPTIMIZER
# -------------------------------------------------
class ATSOptimizer:
    def __init__(self):
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words("english"))

    def extract_keywords(self, text, top_n=20):
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text.lower())
        filtered = [
            w for w in tokens
            if w.isalnum() and w not in self.stop_words and len(w) > 3
        ]
        return [w for w, _ in Counter(filtered).most_common(top_n)]

# -------------------------------------------------
# AI CONTENT GENERATOR
# -------------------------------------------------
class ContentGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _generate(self, prompt, max_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text.replace(prompt, "").strip()

    def generate_summary(self, profile):
        prompt = f"""
Write a concise ATS-optimized professional summary (3–4 sentences).

Name: {profile['name']}
Target Role: {profile['targets']['title']}
Key Skills: {", ".join(profile['skills'][:6])}
"""
        return self._generate(prompt, 180)

    def generate_bullets(self, exp, keywords):
        prompt = f"""
Generate 3–4 resume bullet points.
Start each bullet with • and use ATS keywords.

Role: {exp['title']}
Company: {exp['company']}
Description: {exp['description']}
Keywords: {", ".join(keywords[:8])}
"""
        return self._generate(prompt, 220)

    def generate_cover_letter(self, profile):
        prompt = f"""
Write a professional cover letter.

Candidate: {profile['name']}
Target Role: {profile['targets']['title']}
Company: {profile['targets']['company']}
"""
        return self._generate(prompt, 400)

# -------------------------------------------------
# DOCUMENT GENERATOR
# -------------------------------------------------
class DocumentGenerator:
    def __init__(self):
        self.env = Environment(loader=BaseLoader())
        self.template_path = Path("resume.html")

    def render_html(self, profile, summary, exp_with_bullets):
        template = self.env.from_string(
            self.template_path.read_text(encoding="utf-8")
        )
        return template.render(
            profile=profile,
            summary=summary,
            exp_with_bullets=exp_with_bullets
        )

    def generate_docx(self, profile, summary, exp_with_bullets):
        doc = Document()

        header = doc.add_paragraph(profile["name"])
        header.runs[0].font.size = Pt(24)
        header.runs[0].bold = True
        header.alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_heading("Professional Summary", level=2)
        doc.add_paragraph(summary)

        doc.add_heading("Experience", level=2)
        for exp, bullets in exp_with_bullets:
            p = doc.add_paragraph()
            p.add_run(f"{exp['title']} | {exp['company']}").bold = True
            doc.add_paragraph(bullets)

        doc.add_heading("Education", level=2)
        for edu in profile["education"]:
            p = doc.add_paragraph()
            p.add_run(edu["degree"]).bold = True
            p.add_run(f" | {edu['institution']} | GPA: {edu['gpa']}")

        doc.add_heading("Skills", level=2)
        doc.add_paragraph(", ".join(profile["skills"]))

        path = "resume.docx"
        doc.save(path)
        return path

    def create_zip(self, html, cover_letter, profile):
        zip_name = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(zip_name, "w") as z:
            z.writestr("resume.html", html)
            z.writestr("cover_letter.txt", cover_letter)
            z.writestr("profile.json", json.dumps(profile, indent=2))
        return zip_name

# -------------------------------------------------
# SIDEBAR UI
# -------------------------------------------------
st.sidebar.title("⚙️ Profile Selection")

profile_key = st.sidebar.selectbox(
    "Choose Sample Profile",
    list(SAMPLE_PROFILES.keys()),
    index=0  # Software Engineer default
)

profile = SAMPLE_PROFILES[profile_key]

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.title("🤖 AI Resume & Portfolio Builder")

ats = ATSOptimizer()
generator = ContentGenerator(model, tokenizer)
docgen = DocumentGenerator()

if st.button("✨ Generate Resume & Portfolio"):
    with st.spinner("Generating AI content..."):
        keywords = ats.extract_keywords(profile["targets"]["job_description"])
        summary = generator.generate_summary(profile)

        bullets = [
            generator.generate_bullets(exp, keywords)
            for exp in profile["experience"]
        ]

        cover_letter = generator.generate_cover_letter(profile)
        exp_with_bullets = list(zip(profile["experience"], bullets))

        resume_html = docgen.render_html(profile, summary, exp_with_bullets)
        docx_path = docgen.generate_docx(profile, summary, exp_with_bullets)
        zip_path = docgen.create_zip(resume_html, cover_letter, profile)

    tabs = st.tabs(["📄 Resume Preview", "✉️ Cover Letter", "⬇️ Downloads"])

    with tabs[0]:
        st.components.v1.html(resume_html, height=900, scrolling=True)

    with tabs[1]:
        st.text_area("Cover Letter", cover_letter, height=450)

    with tabs[2]:
        st.download_button("⬇️ Download Resume (HTML)", resume_html, "resume.html")
        st.download_button("⬇️ Download Resume (DOCX)", open(docx_path, "rb"), "resume.docx")
        st.download_button("⬇️ Download Portfolio (ZIP)", open(zip_path, "rb"), zip_path)
