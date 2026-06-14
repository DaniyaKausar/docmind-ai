from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
import re

# Reuse the same embedding model
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def semantic_similarity(list1, list2):
    """
    Real similarity using sentence embeddings + cosine similarity.
    Returns 0-100 score.
    """
    if not list1 or not list2:
        return 0

    model = get_model()
    text1 = " ".join(list1) if isinstance(list1, list) else list1
    text2 = " ".join(list2) if isinstance(list2, list) else list2

    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()
    return round(max(0, score) * 100, 1)

def keyword_match_score(resume_text, jd_keywords):
    """Check what % of JD keywords appear in resume."""
    if not jd_keywords:
        return 0, [], []

    resume_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if kw.lower() in resume_lower]
    missing = [kw for kw in jd_keywords if kw.lower() not in resume_lower]

    score = round((len(matched) / len(jd_keywords)) * 100, 1)
    return score, matched, missing

def calculate_ats_score(resume_data, jd_data, resume_text):
    """
    Calculate comprehensive ATS score using multiple signals.
    All scores are computed, not hardcoded.
    """

    # 1. Skill match (semantic similarity between skill lists)
    skill_score = semantic_similarity(
        resume_data.get("skills", []) + resume_data.get("technologies", []),
        jd_data.get("required_skills", []) + jd_data.get("technologies", [])
    )

    # 2. Keyword match (literal keyword presence)
    all_jd_keywords = (
        jd_data.get("keywords", []) +
        jd_data.get("required_skills", []) +
        jd_data.get("technologies", [])
    )
    keyword_score, matched_kw, missing_kw = keyword_match_score(resume_text, all_jd_keywords)

    # 3. Experience alignment
    resume_exp = resume_data.get("total_experience_years", 0)
    jd_exp_text = jd_data.get("experience_required", "0")
    jd_exp_num = 0
    numbers = re.findall(r'\d+', str(jd_exp_text))
    if numbers:
        jd_exp_num = int(numbers[0])

    if jd_exp_num == 0:
        exp_score = 85.0
    elif resume_exp >= jd_exp_num:
        exp_score = 100.0
    elif resume_exp >= jd_exp_num * 0.7:
        exp_score = 75.0
    elif resume_exp > 0:
        exp_score = 50.0
    else:
        exp_score = 30.0

    # 4. Education alignment (semantic)
    edu_required = jd_data.get("education_required", "")
    resume_edu = " ".join([
        f"{e.get('degree','')} {e.get('field','')}"
        for e in resume_data.get("education", [])
    ])
    edu_score = semantic_similarity(resume_edu, edu_required) if edu_required else 80.0

    # 5. Project relevance (semantic match of projects to JD)
    resume_projects = " ".join([
        f"{p.get('name','')} {p.get('description','')}"
        for p in resume_data.get("projects", [])
    ])
    jd_responsibilities = " ".join(jd_data.get("responsibilities", []))
    project_score = semantic_similarity(resume_projects, jd_responsibilities) if resume_projects else 40.0

    # 6. Overall weighted score
    overall = round(
        skill_score * 0.30 +
        keyword_score * 0.25 +
        exp_score * 0.20 +
        edu_score * 0.10 +
        project_score * 0.15,
        1
    )

    return {
        "overall_score": overall,
        "skill_match": skill_score,
        "keyword_match": keyword_score,
        "experience_alignment": exp_score,
        "education_alignment": edu_score,
        "project_relevance": project_score,
        "matched_keywords": matched_kw[:15],
        "missing_keywords": missing_kw[:15],
        "hiring_readiness": (
            "Strong Candidate" if overall >= 75 else
            "Good Candidate" if overall >= 55 else
            "Needs Improvement"
        )
    }