import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# clean text
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text

# extract skills
skills_db = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "data science",
    "nlp",
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "streamlit",
    "flask",
    "django",
    "html",
    "css",
    "javascript",
    "react",
    "aws",
    "docker",
    "kubernetes",
    "linux",
    "cybersecurity",
    "networking",
    "penetration testing",
    "ethical hacking",
    "wireshark",
    "nmap",
    "burp suite",
    "siem",
    "ids",
    "ips",
    "incident response",
    "vulnerability assessment",
    "owasp",
    "firewall",
    "risk assessment",
    "malware analysis",
    "digital forensics"
]


def extract_skills(text):

    found_skills = []

    for skill in skills_db:
        if skill.lower() in text:
            found_skills.append(skill)

    return found_skills


# resume similarity
def calculate_similarity(resume_text, job_description):

    documents = [resume_text, job_description]

    cv = CountVectorizer()

    matrix = cv.fit_transform(documents)

    similarity = cosine_similarity(matrix)[0][1]

    return round(similarity * 100, 2)


# calculate ats score
def ats_score(similarity, skills_found):

    score = similarity

    if len(skills_found) >= 10:
        score += 10

    elif len(skills_found) >= 5:
        score += 5

    return min(score, 100)


# find missing skills
def missing_skills(job_desc, resume_skills):

    missing = []

    for skill in skills_db:

        if skill in job_desc.lower() and skill not in resume_skills:
            missing.append(skill)

    return missing


# frontend
st.title("AI Resume Screening System")

st.write("Upload Resume PDF and Paste Job Description")

uploaded_resume = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

if st.button("Analyze Resume"):

    if uploaded_resume and job_description:

        # Extract resume text
        resume_text = extract_text_from_pdf(uploaded_resume)

        # Clean
        cleaned_resume = clean_text(resume_text)
        cleaned_jd = clean_text(job_description)

        # Skills
        skills_found = extract_skills(cleaned_resume)

        # Similarity
        similarity_score = calculate_similarity(
            cleaned_resume,
            cleaned_jd
        )

        # ATS
        final_ats = ats_score(
            similarity_score,
            skills_found
        )

        # Missing skills
        missing = missing_skills(
            cleaned_jd,
            skills_found
        )


        # Results

        st.subheader("Results")

        st.write(f"Resume Match Score: {similarity_score}%")

        st.write(f"ATS Score: {final_ats}%")

        st.write("Skills Found:")

        st.success(", ".join(skills_found))

        st.write("Missing Skills:")

        if missing:
            st.error(", ".join(missing))
        else:
            st.success("No major missing skills")

    else:
        st.warning("Please upload resume and enter job description")
