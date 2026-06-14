from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

def get_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.4,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def generate_interview_questions(resume_data, jd_data):
    """Generate targeted interview questions based on resume + JD."""
    llm = get_llm()

    prompt = PromptTemplate.from_template("""
You are a senior technical interviewer at a top tech company.

Candidate Profile:
- Skills: {skills}
- Projects: {projects}
- Experience: {experience}

Target Role: {role} at {company}
Required Skills: {required_skills}
Responsibilities: {responsibilities}

Generate a comprehensive interview preparation guide:

## 🔧 Technical Questions (Based on JD requirements)
**Easy:**
1. [Question] → Key answer points

**Medium:**
2. [Question] → Key answer points
3. [Question] → Key answer points

**Hard:**
4. [Question] → Key answer points

## 💼 Project-Based Questions (Based on candidate's projects)
1. [Specific question about their actual project]
   → What interviewer wants to hear

2. [Question about technical decisions]
   → What interviewer wants to hear

## 🧠 Behavioral Questions (STAR method)
1. [Question] → Structure your answer around: [hint]
2. [Question] → Structure your answer around: [hint]
3. [Question] → Structure your answer around: [hint]

## 🤝 HR Questions
1. [Question] → Key talking points
2. [Question] → Key talking points

## ⚡ Questions YOU Should Ask the Interviewer
1. [Smart question that shows research]
2. [Question about growth/team]
3. [Technical question about their stack]

## 🎯 Red Flags to Avoid
[2-3 common mistakes candidates make for this specific role]
""")

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "skills": ", ".join(resume_data.get("skills", [])[:15]),
        "projects": ", ".join([p.get("name", "") for p in resume_data.get("projects", [])[:3]]),
        "experience": resume_data.get("total_experience_years", 0),
        "role": jd_data.get("role", "Software Engineer"),
        "company": jd_data.get("company", "the company"),
        "required_skills": ", ".join(jd_data.get("required_skills", [])[:10]),
        "responsibilities": ", ".join(jd_data.get("responsibilities", [])[:5])
    })