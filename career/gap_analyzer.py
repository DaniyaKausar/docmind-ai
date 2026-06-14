from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, util
import os

def get_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def find_missing_skills(resume_data, jd_data):
    """Find skills in JD not present in resume using semantic similarity."""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    resume_skills = resume_data.get("skills", []) + resume_data.get("technologies", [])
    jd_required = jd_data.get("required_skills", []) + jd_data.get("technologies", [])
    jd_preferred = jd_data.get("preferred_skills", [])

    missing_required = []
    missing_preferred = []

    for skill in jd_required:
        if not resume_skills:
            missing_required.append(skill)
            continue
        skill_emb = model.encode(skill, convert_to_tensor=True)
        resume_embs = model.encode(resume_skills, convert_to_tensor=True)
        scores = util.cos_sim(skill_emb, resume_embs)[0]
        if scores.max().item() < 0.65:  # threshold for "missing"
            missing_required.append(skill)

    for skill in jd_preferred:
        if not resume_skills:
            missing_preferred.append(skill)
            continue
        skill_emb = model.encode(skill, convert_to_tensor=True)
        resume_embs = model.encode(resume_skills, convert_to_tensor=True)
        scores = util.cos_sim(skill_emb, resume_embs)[0]
        if scores.max().item() < 0.65:
            missing_preferred.append(skill)

    return missing_required, missing_preferred


def generate_learning_roadmap(resume_data, jd_data, missing_required, missing_preferred):
    """Generate a prioritized learning roadmap using LLM."""
    llm = get_llm()

    prompt = PromptTemplate.from_template("""
You are a career coach. Based on this candidate's profile and the job requirements, 
create a specific, actionable learning roadmap.

Candidate Skills: {resume_skills}
Target Role: {role}
Missing Required Skills: {missing_required}
Missing Preferred Skills: {missing_preferred}

Generate a roadmap with this structure:

## 🔴 High Priority (Learn in 2-4 weeks)
[Skills that are required and missing — list each with a specific free resource]

## 🟡 Medium Priority (Learn in 1-2 months)  
[Skills that are preferred or would strengthen the profile]

## 🟢 Low Priority (Nice to have)
[Additional skills that would make the candidate stand out]

## ⏱️ Estimated Timeline
[Realistic timeline to be job-ready]

## 🎯 Quick Wins
[2-3 things candidate can do THIS WEEK to improve their chances]

Be specific. Mention actual courses, platforms (Coursera, YouTube, docs).
""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "resume_skills": ", ".join(resume_data.get("skills", [])[:20]),
        "role": jd_data.get("role", "the target role"),
        "missing_required": ", ".join(missing_required[:10]) or "None",
        "missing_preferred": ", ".join(missing_preferred[:10]) or "None"
    })


def generate_resume_improvements(resume_data, jd_data, ats_scores):
    """Generate specific resume improvement suggestions."""
    llm = get_llm()

    prompt = PromptTemplate.from_template("""
You are an expert resume reviewer and ATS optimization specialist.

Resume Summary: {resume_summary}
Target Role: {role}
ATS Score: {ats_score}/100
Missing Keywords: {missing_keywords}

Generate specific, actionable improvements:

## ✅ Resume Strengths
[What's working well — be specific]

## ❌ Critical Gaps
[What's missing or weak — be direct]

## 🔧 ATS Optimization Tips
[Exact keywords to add and where to add them]

## 📝 Bullet Point Improvements
[2-3 example rewrites of weak bullet points using the STAR method with numbers]

## 🏆 One Thing To Do Today
[Single highest-impact action]
""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "resume_summary": resume_data.get("summary", ""),
        "role": jd_data.get("role", ""),
        "ats_score": ats_scores.get("overall_score", 0),
        "missing_keywords": ", ".join(ats_scores.get("missing_keywords", [])[:10])
    })