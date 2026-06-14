from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
import re

def get_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def clean_json(text):
    """Extract JSON from LLM response safely."""
    # Find JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {}

def extract_resume_data(resume_text):
    """Extract structured data from resume text."""
    llm = get_llm()

    prompt = PromptTemplate.from_template("""
Extract information from this resume and return ONLY a JSON object.
No explanation, no markdown, just raw JSON.

Resume:
{text}

Return this exact JSON structure:
{{
  "name": "candidate name or Unknown",
  "email": "email or empty string",
  "skills": ["skill1", "skill2"],
  "technologies": ["tech1", "tech2"],
  "education": [{{"degree": "", "field": "", "institution": "", "year": ""}}],
  "experience": [{{"title": "", "company": "", "duration": "", "description": ""}}],
  "projects": [{{"name": "", "description": "", "technologies": []}}],
  "certifications": ["cert1"],
  "total_experience_years": 0,
  "summary": "2 sentence professional summary"
}}
""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": resume_text[:6000]})
    return clean_json(result)


def extract_jd_data(jd_text):
    """Extract structured data from job description."""
    llm = get_llm()

    prompt = PromptTemplate.from_template("""
Extract information from this job description and return ONLY a JSON object.
No explanation, no markdown, just raw JSON.

Job Description:
{text}

Return this exact JSON structure:
{{
  "role": "job title",
  "company": "company name or Unknown",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "technologies": ["tech1", "tech2"],
  "responsibilities": ["resp1", "resp2"],
  "education_required": "degree requirement",
  "experience_required": "X years",
  "keywords": ["keyword1", "keyword2"],
  "summary": "2 sentence job summary"
}}
""")

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": jd_text[:6000]})
    return clean_json(result)