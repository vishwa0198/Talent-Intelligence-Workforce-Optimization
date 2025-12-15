import os
import json
import time
from typing import List, Dict, Optional

# For file handling + PDF/resume text extraction
import fitz  # PyMuPDF
from pathlib import Path

# OpenAI >= 1.0.0
from openai import OpenAI

# Optional: sentence-transformers (only if you want additional embedding-based tasks)
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTE = True
except Exception:
    _HAS_SENTE = False

# Optionally load variables from a .env file (if python-dotenv is installed)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # Safe to ignore; environment variables may already be set at OS level.
    pass

# Load OPENAI_API_KEY from env (or .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # For Streamlit, it's nicer to fail lazily; log a warning instead of raising immediately.
    print("WARNING: OPENAI_API_KEY not set. LLM features in chatbot will not work until it is configured.")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# Default model (you can change)
DEFAULT_MODEL = "gpt-3.5-turbo"           # change if you prefer "gpt-4", "gpt-3.5-turbo", etc.
DEFAULT_TEMPERATURE = 0.3

# ---- Utilities ----
def safe_read_pdf_bytes(file_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF (fitz).
    Returns extracted text (empty string on failure).
    """
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print("PDF text extraction failed:", e)
    return text

def safe_read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ---- OpenAI wrapper ----
def call_openai_chat(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 1024,
    retry: int = 2,
) -> str:
    """
    Call OpenAI ChatCompletion with a list of messages (OpenAI >= 1.0.0).
    messages = [{"role":"system","content":...}, {"role":"user","content":...}, ...]
    Returns assistant content as string.
    """
    if client is None:
        raise RuntimeError("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
    
    last_exc = None
    for attempt in range(retry + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Response structure for OpenAI >= 1.0.0
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt * 2)
    raise RuntimeError(f"OpenAI call failed after retries: {last_exc}")

# ---- High-level HR LLM functions ----

def chat_with_hr_bot(
    user_message: str,
    history: Optional[List[Dict[str,str]]] = None,
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Dict[str, object]:
    """
    Single-turn chatbot interface. Provide a system prompt or use a default HR assistant prompt.
    history: list of {"role": "user"|"assistant", "content": "..."} entries to preserve context.
    Returns {"response": str, "messages": [...]}
    """
    if system_prompt is None:
        system_prompt = (
            "You are an HR assistant for a mid-size company. "
            "Answer clearly and concisely. If giving procedures, number steps. "
            "If policy/regulatory questions might require legal help, say so politely. "
            "Be professional and concise."
        )
    messages = [{"role":"system","content":system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role":"user","content":user_message})

    assistant_text = call_openai_chat(messages, model=model, temperature=temperature)
    return {"response": assistant_text, "messages": messages + [{"role":"assistant","content":assistant_text}]}


def summarize_interview_transcript(
    transcript_text: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    summary_length: str = "short",
) -> str:
    """
    Summarize interview transcript and extract: top 5 skills, strengths, weaknesses, recommended hire (yes/no) with reasoning.
    summary_length: "short"|"detailed"
    """
    if not transcript_text or len(transcript_text.strip()) == 0:
        return "Transcript empty."

    if summary_length == "short":
        instruct = (
            "Create a short interview summary (5-7 bullets). Then list top 5 skills mentioned. "
            "Provide 3 strengths, 3 weaknesses, and a short hiring recommendation (Yes/No + 1-sentence reason)."
        )
    else:
        instruct = (
            "Create a detailed interview summary with sections: Overview, Key Skills, Strengths, Weaknesses, "
            "Red flags (if any), and a hiring recommendation with pros and cons. Be thorough but concise."
        )

    messages = [
        {"role":"system","content":"You are an expert hiring manager and interviewer analyst."},
        {"role":"user","content":f"Transcript:\n\n{transcript_text}\n\n{instruct}"}
    ]

    return call_openai_chat(messages, model=model, temperature=temperature, max_tokens=1200)


def generate_job_description(
    role_name: str,
    skills_bullets: str,
    seniority: str = "Mid",
    responsibilities: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    """
    Generate a professional job description from role name + skill list.
    skills_bullets: multiline string or comma-separated skills.
    responsibilities: optional list of responsibilities (if None, LLM will produce).
    """
    prompt = (
        f"Generate a clean, ATS-friendly job description for role: {role_name}. "
        f"Seniority: {seniority}. Skills: {skills_bullets}. "
        f"Format: Job Title, Summary (2 sentences), Responsibilities (bullet list), Required Qualifications (bullet list), Nice-to-have Skills (bullet list), Salary range example (optional)."
    )
    if responsibilities:
        prompt += "\nInclude the following responsibilities: " + "; ".join(responsibilities)

    messages = [
        {"role":"system","content":"You are a recruiting specialist that writes clear, concise, and structured job descriptions."},
        {"role":"user","content":prompt}
    ]

    return call_openai_chat(messages, model=model, temperature=temperature, max_tokens=700)


def resume_improvement_feedback(
    resume_text: str,
    role_target: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0
) -> str:
    """
    Provide feedback on a resume: strengths, weaknesses, missing keywords for a target role, suggested bullet re-writes, and a short ATS score.
    """
    if not resume_text:
        return "No resume text provided."

    prompt = "You are an expert recruiter. "
    if role_target:
        prompt += f"Candidate aims for role: {role_target}. "
    prompt += (
        "Provide:\n"
        "1) A short summary of the resume (2 sentences).\n"
        "2) Strengths (3 bullets).\n"
        "3) Weaknesses / gaps (3 bullets).\n"
        "4) Missing keywords / skills for the target role (if role provided).\n"
        "5) Rewrite 3 sample bullets to be stronger (show before -> after format).\n"
        "6) Give an ATS-friendliness score from 0-100 and suggest 3 improvements to improve score.\n"
    )

    messages = [
        {"role":"system","content":"You are a senior recruiter who writes clear, actionable resume feedback."},
        {"role":"user","content":prompt + "\n\nResume:\n\n" + resume_text}
    ]

    return call_openai_chat(messages, model=model, temperature=temperature, max_tokens=1200)


def analyze_employee_review(
    review_text: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0
) -> str:
    """
    Given a review text (like Glassdoor review), return sentiment, summary, actionable HR insights (3 bullets), and a suggested response to the reviewer.
    """
    messages = [
        {"role":"system","content":"You are an HR analyst that converts employee reviews into actionable insights."},
        {"role":"user","content":(
            "Analyze this employee review. Return: 1) sentiment (Positive/Neutral/Negative), 2) 2-3 sentence summary, "
            "3) 3 actionable recommendations HR can take, and 4) draft professional response to the reviewer (2-3 sentences).\n\n"
            f"Review:\n{review_text}"
        )}
    ]
    return call_openai_chat(messages, model=model, temperature=temperature, max_tokens=700)

# ---- Example CLI / test runner ----
if __name__ == "__main__":
    print("chatbot.py - Basic smoke test (interactive)")

    # Simple interactive CLI for quick testing
    while True:
        print("\n[Options] 1) Chat 2) Summarize transcript 3) Resume feedback 4) Generate JD 5) Analyze review 6) Exit")
        choice = input("Select option: ").strip()
        if choice == "1":
            msg = input("Type your HR question: ").strip()
            out = chat_with_hr_bot(msg)
            print("\n--- Assistant ---\n", out["response"])
        elif choice == "2":
            path = input("Path to transcript (txt or pdf): ").strip()
            if path.lower().endswith(".pdf"):
                txt = safe_read_pdf_bytes(path)
            else:
                txt = safe_read_txt(path)
            print("\n--- Summary ---\n")
            print(summarize_interview_transcript(txt))
        elif choice == "3":
            path = input("Path to resume (txt or pdf): ").strip()
            if path.lower().endswith(".pdf"):
                txt = safe_read_pdf_bytes(path)
            else:
                txt = safe_read_txt(path)
            role = input("Target role (optional): ").strip()
            print("\n--- Resume Feedback ---\n")
            print(resume_improvement_feedback(txt, role_target=role if role else None))
        elif choice == "4":
            role = input("Role name: ").strip()
            skills = input("Skills (comma separated or bullets): ").strip()
            seniority = input("Seniority (Junior/Mid/Senior): ").strip() or "Mid"
            print("\n--- Job Description ---\n")
            print(generate_job_description(role, skills, seniority=seniority))
        elif choice == "5":
            review = input("Paste short review text: ").strip()
            print("\n--- Review Analysis ---\n")
            print(analyze_employee_review(review))
        elif choice == "6":
            print("Exiting.")
            break
        else:
            print("Unknown option. Choose 1-6.")
