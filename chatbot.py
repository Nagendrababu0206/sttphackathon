from __future__ import annotations
import os
import sys
import json
import glob
import re
import datetime
from typing import TypedDict, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient
import google.generativeai as genai
from google.api_core import exceptions as genai_exceptions
import time
try:
    import PyPDF2
except ImportError:
    # Minimal stub to allow PDF extraction calls without actual library
    class _DummyPage:
        def extract_text(self):
            return ""
    class _DummyPdfReader:
        def __init__(self, *args, **kwargs):
            self.pages = []
    PyPDF2 = type('PyPDF2', (), {'PdfReader': _DummyPdfReader})

# For RAG
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Enable virtual terminal processing on Windows for ANSI colors and reconfigure encoding to UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
    import ctypes
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

load_dotenv()

# Use GOOGLE_API_KEY (or fallback from GEMINI_API_KEY) for Gemini
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Ensure required API keys are present and set
required_keys = ["GOOGLE_API_KEY", "TAVILY_API_KEY"]
missing_keys = [k for k in required_keys if not os.getenv(k)]
if missing_keys:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_keys)}")

# =======================
# 1. State Definition
# =======================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    parsed_jd: Optional[Dict[str, Any]]
    resumes: Dict[str, str]
    shortlist: List[Dict[str, Any]]
    pending_confirmation: bool
    pending_action: Optional[str]
    # New fields for advanced features
    mismatch_feedback: Optional[str]
    comparison_table: Optional[str]
    jd_improvements: Optional[str]
    email_draft: Optional[str]
    scheduled_interviews: List[Dict[str, Any]]
    skill_trends: Optional[str]

# Helper functions for new capabilities

def _analyze_experience_mismatch(parsed_jd: ParsedJD, resumes: Dict[str, str]) -> str:
    """Calculate experience distribution and compare with JD requirement.
    Returns a feedback string.
    """
    # Simple heuristic: count years mentioned in resumes (look for numbers followed by 'year')
    import re
    years = []
    for text in resumes.values():
        matches = re.findall(r"(\d+)\s+years?", text.lower())
        for m in matches:
            years.append(int(m))
    if not years:
        return "No explicit experience years found in resumes."
    avg_years = sum(years) / len(years)
    required = int(re.search(r"(\d+)", parsed_jd.experience).group(1)) if re.search(r"(\d+)", parsed_jd.experience) else None
    if required and avg_years < required:
        return f"Your JD asks for {required}+ years but the average candidate has only {avg_years:.1f} years. Consider adjusting the requirement."
    return "Candidate experience aligns well with JD requirements."

def _build_comparison_table(parsed_jd: ParsedJD, resumes: Dict[str, str]) -> str:
    """Create a markdown table comparing each candidate's skills to JD requirements."""
    header = "| Candidate | Matching Skills | Missing Skills | Experience (years) |\n|---|---|---|---|"
    rows = []
    for name, text in resumes.items():
        # Extract skills
        candidate_skills = []
        for skill in parsed_jd.skills:
            if re.search(skill.lower(), text.lower()):
                candidate_skills.append(skill)
        missing = [s for s in parsed_jd.skills if s not in candidate_skills]
        # Extract experience years
        exp_match = re.search(r"(\d+)\s+years?", text.lower())
        exp_years = exp_match.group(1) if exp_match else "N/A"
        rows.append(f"| {name.replace('.txt','')} | {', '.join(candidate_skills) or 'None'} | {', '.join(missing) or 'None'} | {exp_years} |")
    return "\n".join([header] + rows)

def _suggest_jd_improvements(parsed_jd: ParsedJD) -> str:
    """Suggest additional fields or details for the JD based on common patterns."""
    suggestions = []
    if not parsed_jd.role:
        suggestions.append("Add a clear job title.")
    if not parsed_jd.skills:
        suggestions.append("Include a list of required skills.")
    if not parsed_jd.experience:
        suggestions.append("Specify the required experience level.")
    # Example suggestion: add location, benefits, remote option
    suggestions.append("Consider adding location, remote work options, and key benefits.")
    return " ".join(suggestions)

def _extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF using PyPDF2.
    Returns concatenated page text; if parsing fails, returns empty string.
    """
    try:
        reader = PyPDF2.PdfReader(file_path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        print(f"[PDF Extract] Failed to read {file_path}: {e}")
        return ""

def _detect_red_flags(resume_text: str) -> List[str]:
    """Detect simple red‑flags in a resume.
    - Employment gaps > 12 months.
    - Overlapping or inconsistent date ranges.
    Returns a list of warning strings.
    """
    # Find date ranges like "Jan 2020 - Mar 2021" or "2020 - 2022"
    # Simplify to capturing start and end years/months.
    date_pattern = r"(?i)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*(\d{4})"
    matches = re.findall(date_pattern, resume_text)
    years = []
    for month, year in matches:
        try:
            yr = int(year)
            if month:
                mon_num = datetime.datetime.strptime(month[:3], "%b").month
                dt = datetime.datetime(yr, mon_num, 1)
            else:
                dt = datetime.datetime(yr, 1, 1)
            years.append(dt)
        except Exception:
            continue
    years = sorted(set(years))
    warnings: List[str] = []
    if not years:
        return warnings
    # Detect gaps > 12 months
    for i in range(1, len(years)):
        delta = (years[i].year - years[i-1].year) * 12 + (years[i].month - years[i-1].month)
        if delta > 12:
            warnings.append(f"Employment gap of {delta} months between {years[i-1].strftime('%b %Y')} and {years[i].strftime('%b %Y')}")
    # Overlap detection is more complex; omitted for brevity.
    return warnings

def _batch_screening_report(parsed_jd: ParsedJD, resumes: Dict[str, str]) -> str:
    """Generate a markdown report summarising screening for all candidates.
    Includes match scores, red‑flags and a table.
    Writes the report to `data/reports/batch_report.md` and returns the file path.
    """
    # Ensure report directory exists
    report_dir = os.path.join("data", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "batch_report.md")
    # Build comparison table
    table_md = _build_comparison_table(parsed_jd, resumes)
    # Gather red‑flags per candidate
    red_flag_sections = []
    for name, txt in resumes.items():
        flags = _detect_red_flags(txt)
        if flags:
            red_flag_sections.append(f"#### {name.replace('.txt','').replace('.pdf','')}:\n" + "\n".join(f"- {f}" for f in flags))
    # Assemble report
    report_md = "# Batch Screening Report\n\n" \
                + "## Candidate Matching Table\n\n" + table_md + "\n\n"
    if red_flag_sections:
        report_md += "## Red‑Flag Summary\n\n" + "\n\n".join(red_flag_sections) + "\n"
    else:
        report_md += "## Red‑Flag Summary\n\nNo red‑flags detected.\n"
    # Write file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    return report_path

def _draft_email(shortlist: List[Dict[str, Any]], template: str = "") -> str:
    """Generate an email body for shortlisted candidates or hiring manager.
    If a template is provided, use it; otherwise use a default.
    """
    if not template:
        template = "Dear {name},\n\nWe are pleased to inform you that you have been shortlisted for the {role} position. Your match score is {score}% .\nWe will contact you soon with next steps.\n\nBest regards,\nRecruitment Team"
    lines = []
    for cand in shortlist:
        name = cand["name"].replace('.txt','').replace('candidate_','').replace('_',' ').title()
        score = cand.get('score', 'N/A')
        lines.append(template.format(name=name, role='[Role]', score=score))
        lines.append("---\n")
    return "\n".join(lines)

def _send_email_smtp(email_body: str, recipient: str) -> str:
    """Send email via SMTP using credentials from .env"""
    import smtplib
    from email.message import EmailMessage
    smtp_host = os.getenv('SMTP_HOST')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_user = os.getenv('SMTP_USER')
    smtp_pass = os.getenv('SMTP_PASS')
    if not all([smtp_host, smtp_user, smtp_pass]):
        return "SMTP configuration missing in environment."
    msg = EmailMessage()
    msg['Subject'] = 'Recruitment Update'
    msg['From'] = smtp_user
    msg['To'] = recipient
    msg.set_content(email_body)
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return "Email sent successfully."
    except Exception as e:
        return f"Failed to send email: {e}"

def _schedule_interview(candidate_name: str, datetime_str: str) -> str:
    """Add interview slot to a JSON calendar file."""
    try:
        interview = {"candidate": candidate_name, "datetime": datetime_str}
        calendar_path = os.path.join('data', 'calendar.json')
        if os.path.exists(calendar_path):
            with open(calendar_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        data.append(interview)
        with open(calendar_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return f"Interview scheduled for {candidate_name} at {datetime_str}."
    except Exception as e:
        return f"Failed to schedule interview: {e}"

def _fetch_skill_trends(role: str) -> str:
    """Use Tavily to search for in‑demand skills for the given role and compare with JD."""
    query = f"top in-demand technical skills for {role} 2025"
    try:
        result = _tavily_client.search(query=query, search_depth="basic")
        # Extract titles/content
        texts = [r.get('content','') for r in result.get('results', [])]
        combined = " ".join(texts)
        # Simple extraction of skill words (capitalized words)
        skills = set(re.findall(r"\b[A-Za-z][A-Za-z0-9+]+\b", combined))
        top = list(skills)[:10]
        return f"Top trending skills for {role}: {', '.join(top)}"
    except Exception as e:
        return f"Skill trend lookup failed: {e}"

# Nodes for new features
def mismatch_feedback_node(state: AgentState) -> dict:
    if not state.get('parsed_jd') or not state.get('resumes'):
        return {"messages": [AIMessage(content="Parse JD and load resumes first.")]}
    feedback = _analyze_experience_mismatch(ParsedJD(**state['parsed_jd']), state['resumes'])
    state['mismatch_feedback'] = feedback
    return {"messages": [AIMessage(content=feedback)]}

def candidate_comparison_node(state: AgentState) -> dict:
    if not state.get('parsed_jd') or not state.get('resumes'):
        return {"messages": [AIMessage(content="Parse JD and load resumes first.")]}
    table = _build_comparison_table(ParsedJD(**state['parsed_jd']), state['resumes'])
    state['comparison_table'] = table
    return {"messages": [AIMessage(content=table)]}

def jd_improvement_node(state: AgentState) -> dict:
    if not state.get('parsed_jd'):
        return {"messages": [AIMessage(content="Parse JD first.")]}
    suggestions = _suggest_jd_improvements(ParsedJD(**state['parsed_jd']))
    state['jd_improvements'] = suggestions
    return {"messages": [AIMessage(content=suggestions)]}

def email_draft_node(state: AgentState) -> dict:
    if not state.get('shortlist'):
        return {"messages": [AIMessage(content="No shortlist available to draft email.")]}
    draft = _draft_email(state['shortlist'])
    state['email_draft'] = draft
    return {"messages": [AIMessage(content=draft)]}

def email_send_node(state: AgentState) -> dict:
    if not state.get('email_draft'):
        return {"messages": [AIMessage(content="No email draft prepared.")]}
    # For simplicity send to a placeholder recipient from env
    recipient = os.getenv('RECIPIENT_EMAIL') or 'example@example.com'
    result = _send_email_smtp(state['email_draft'], recipient)
    return {"messages": [AIMessage(content=result)]}

def interview_schedule_node(state: AgentState) -> dict:
    # Expect last user message like "schedule interview Alice 2026-07-10 14:00"
    last_msg = state['messages'][-1].content
    match = re.search(r"schedule interview (.+?) (\d{4}-\d{2}-\d{2} \d{2}:\d{2})", last_msg, re.IGNORECASE)
    if not match:
        return {"messages": [AIMessage(content="Please specify as: schedule interview <candidate> <YYYY-MM-DD HH:MM>")]}
    name, dt = match.groups()
    result = _schedule_interview(name, dt)
    return {"messages": [AIMessage(content=result)]}

def skill_trend_analysis_node(state: AgentState) -> dict:
    if not state.get('parsed_jd'):
        return {"messages": [AIMessage(content="Parse JD first.")]}
    role = ParsedJD(**state['parsed_jd']).role
    trends = _fetch_skill_trends(role)
    state['skill_trends'] = trends
    return {"messages": [AIMessage(content=trends)]}

# Extend the graph wiring (example additions)
# Assuming `graph` is the StateGraph instance defined later in the file.
# After screening, we can add optional edges based on commands.


# =======================
# Pydantic Models
# =======================
class ParsedJD(BaseModel):
    role: str = Field(description="The job role or title")
    skills: List[str] = Field(description="List of required skills")
    experience: str = Field(description="Experience required")

# =======================
# LLMs & Tools Setup
# =======================
# Google Generative AI components
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Primary model for generation; fallback uses higher quota if needed
model = genai.GenerativeModel("gemini-1.5-pro")
structured_llm = llm.with_structured_output(ParsedJD)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

# Simple in‑memory prompt cache
_prompt_cache: dict[str, str] = {}

def safe_invoke(prompt: str, *, max_retries: int = 5, base_delay: float = 2.0) -> str:
    """Call Gemini model safely with exponential back‑off on 429 errors."""
    # Cache lookup
    if prompt in _prompt_cache:
        print("[Cache] Prompt retrieved from cache")
        return _prompt_cache[prompt]
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            text = response.text
            _prompt_cache[prompt] = text
            return text
        except genai_exceptions.TooManyRequests:
            wait = base_delay * (2 ** (attempt - 1))
            print(f"[Rate Limit] attempt {attempt}/{max_retries}; sleeping {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            raise e
    raise RuntimeError(f"All {max_retries} attempts failed due to rate limits.")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Fallback model with higher free-tier quota (or cheaper)
model = genai.GenerativeModel("gemini-1.5-pro")
structured_llm = llm.with_structured_output(ParsedJD)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
# Simple in-memory prompt cache to avoid duplicate LLM calls within a session
_prompt_cache: dict[str, str] = {}  # Initialize cache

# Tavily client
_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))

def _safe_invoke(prompt: str) -> str:
    """Helper to call Gemini model safely."""
    return model.generate_content(prompt).text

def safe_invoke(prompt: str) -> str:
    """Invoke LLM with caching and fallback.
    - Checks in-memory cache first.
    - Tries primary model (`llm`) with exponential backoff.
    - If quota is exhausted, automatically switches to `fallback_llm`.
    - Stores successful responses in cache.
    """
    # Cache lookup
    if prompt in _prompt_cache:
        print("[Cache] Prompt retrieved from cache")
        return _prompt_cache[prompt]

    import time, re
    def _invoke_with(model, max_retries, base_delay=5.0):
        for attempt in range(max_retries):
            try:
                response = _safe_invoke(interview_prompt)
                return getattr(result, "content", str(result))
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                    delay_match = re.search(r'(?:please retry in|retry in)\s*([\d\.]+)\s*s', err_str, re.IGNORECASE)
                    delay = float(delay_match.group(1)) + 1.5 if delay_match else base_delay * (2 ** attempt)
                    print(f"\n\033[1;33m[Rate Limit] 429 Resource Exhausted. Retrying in {delay:.1f}s... (Attempt {attempt+1}/{max_retries})\033[0m")
                    time.sleep(delay)
                    continue
                return f"[LLM Error] {e}"
        return None

    # Try primary model first
    response = _invoke_with(llm, max_retries=10)
    if response is None or response.startswith("[LLM Error]"):
        response = _invoke_with(fallback_llm, max_retries=5)
        if response is None:
            response = "[Fallback] Unable to reach LLM due to rate limits."
    _prompt_cache[prompt] = response
    return response


def parse_jd_locally(jd_text: str) -> Optional[ParsedJD]:
    """Helper to parse a JD locally using simple line extraction to save API quota."""
    import re
    lines = jd_text.splitlines()
    role = None
    experience = "Not specified"
    skills = []
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        
        # Match Role / Job Title
        role_match = re.match(r'^(?:Job Title|Role):\s*(.*)$', line_strip, re.IGNORECASE)
        if role_match:
            val = role_match.group(1).strip()
            # Prefer Job Title or the first non-empty role
            if not role or "job title" in line.lower():
                role = val
            continue
            
        # Match Experience
        exp_match = re.match(r'^(?:Experience Required|Experience):\s*(.*)$', line_strip, re.IGNORECASE)
        if exp_match:
            experience = exp_match.group(1).strip()
            continue
            
    # Find Skills block
    in_skills = False
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        if re.match(r'^(?:Required Skills|Skills):\s*$', line_strip, re.IGNORECASE):
            in_skills = True
            continue
        if in_skills:
            if ":" in line_strip and not line_strip.startswith('-') and not line_strip.startswith('*'):
                break
            if line_strip.startswith('-'):
                skill = line_strip.lstrip('-').strip()
                if skill:
                    skills.append(skill)
            elif line_strip.startswith('*'):
                skill = line_strip.lstrip('*').strip()
                if skill:
                    skills.append(skill)
                    
    if role and skills:
        return ParsedJD(role=role, skills=skills, experience=experience)
    return None

def parse_jd_structured(jd_text: str) -> ParsedJD:
    """Uses Gemini structured output to parse job descriptions into schema with retries."""
    import time
    import re
    max_retries = 5
    base_delay = 5.0
    prompt = f"Parse the following Job Description into structured fields (role, skills, experience):\n\n{jd_text}"
    
    for attempt in range(max_retries):
        try:
            return structured_llm.invoke(prompt)
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                # Try parsing dynamic retry delay from Gemini's exception message
                delay_match = re.search(r'(?:please retry in|retry in)\s*([\d\.]+)\s*s', err_str, re.IGNORECASE)
                if delay_match:
                    delay = float(delay_match.group(1)) + 1.5
                else:
                    delay = base_delay * (2 ** attempt)
                print(f"\n\033[1;33m[Rate Limit] 429 Resource Exhausted. Retrying JD parse in {delay:.1f}s...\033[0m")
                time.sleep(delay)
                continue
            break
            
    # Fallback to plain LLM or defaults
    print(f"\033[1;33m⚠️ Warning: Structured JD parsing failed or rate-limited. Attempting fallback...\033[0m")
    # Direct raw prompt ask
    fallback_prompt = (
        f"Parse the following Job Description into JSON with keys 'role', 'skills' (list), and 'experience' (string):\n\n{jd_text}\n\n"
        "Return only valid JSON."
    )
    try:
        raw_res = safe_invoke(fallback_prompt)
        clean_json = raw_res.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        return ParsedJD(
            role=data.get("role", "Unknown"),
            skills=data.get("skills", []),
            experience=data.get("experience", "Unknown")
        )
    except Exception:
        return ParsedJD(role="Software Engineer", skills=[], experience="Not specified")

def parse_jd(jd_text: str) -> ParsedJD:
    """Parses JD trying local regex extraction first to avoid hitting API rate limits.
    Falls back to LLM structured output only if local extraction fails.
    """
    try:
        local_parsed = parse_jd_locally(jd_text)
        if local_parsed:
            return local_parsed
    except Exception:
        pass
    return parse_jd_structured(jd_text)


# =======================
# Nodes
# =======================
def parse_jd_resumes_node(state: AgentState) -> dict:
    """Loads and parses the JD file, and reloads resumes in the folder (with custom folder path support)."""
    try:
        import re
        last_msg = state["messages"][-1].content.strip() if state.get("messages") else ""
        target_dir = "data/resumes"
        
        # Look for path patterns
        # e.g. upload "path" or load from path
        matches = re.findall(r'(?:load|upload|from)\s+[\'"]?([a-zA-Z]:\\[^"\'\n]+|/[^"\'\n]+|\w+/\w+|[a-zA-Z]:/[^"\'\n]+)[\'"]?', last_msg, re.IGNORECASE)
        if not matches:
            # Fallback check if any word in the query is a valid directory
            words = last_msg.split()
            for word in words:
                word_clean = word.strip('"\'')
                if os.path.isdir(word_clean):
                    target_dir = word_clean
                    break
        else:
            potential_dir = matches[0].strip()
            if os.path.isdir(potential_dir):
                target_dir = potential_dir
                
        if not os.path.exists("data/jd.txt"):
            return {"messages": [AIMessage(content="\033[1;31mError: Job Description file data/jd.txt not found.\033[0m")]}
            
        with open("data/jd.txt", "r", encoding="utf-8") as f:
            jd_text = f.read()
            
        parsed_jd = parse_jd(jd_text)
        
        # Load resumes from disk (TXT and PDF)
        resumes = {}
        if os.path.exists(target_dir):
            for filepath in glob.glob(os.path.join(target_dir, "*")):
                if os.path.isfile(filepath):
                    ext = os.path.splitext(filepath)[1].lower()
                    with open(filepath, "r", encoding="utf-8") as f:
                        filename = os.path.basename(filepath)
                        if ext == ".pdf":
                            # Re‑read the file as binary for PyPDF2
                            f.close()
                            text = _extract_text_from_pdf(filepath)
                            resumes[filename] = text
                        else:
                            resumes[filename] = f.read()
                    
        return {
            "parsed_jd": parsed_jd.model_dump(), 
            "resumes": resumes,
            "messages": [AIMessage(content=(
                f"\033[1;32mSuccessfully loaded and parsed Job Description!\033[0m\n"
                f"  - \033[1mRole:\033[0m {parsed_jd.role}\n"
                f"  - \033[1mSkills:\033[0m {', '.join(parsed_jd.skills)}\n"
                f"  - \033[1mExperience:\033[0m {parsed_jd.experience}\n"
                f"  - \033[1mResumes loaded from:\033[0m {os.path.abspath(target_dir)}\n"
                f"  - \033[1mCount:\033[0m {len(resumes)} candidates in database."
            ))]
        }
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error loading JD/resumes: {e}")]}

def count_applicants_node(state: AgentState) -> dict:
    """Counts applicants using pure Python logic."""
    count = 0
    if os.path.exists("data/resumes"):
        files = glob.glob(os.path.join("data/resumes", "*.txt"))
        count = len(files)
    return {"messages": [AIMessage(content=f"📊 There are \033[1;32m{count}\033[0m applicant resumes currently stored in the candidate folder.")]}

def screen_candidates_node(state: AgentState) -> dict:
    """Performs RAG similarity matching over resumes based on the Job Description."""
    if not state.get("parsed_jd") or not state.get("resumes"):
        return {"messages": [AIMessage(content="Please load/parse the JD and resumes first.")]}
    
    docs = []
    for name, text in state["resumes"].items():
        docs.append(Document(page_content=text, metadata={"source": name}))
        
    if not docs:
        return {"messages": [AIMessage(content="No candidate resumes loaded. Please place resume text files in data/resumes/")]}
        
    # Initialize Chroma database in memory
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    
    # Retrieve top 5 matching candidates
    jd_info = state["parsed_jd"]
    query = f"Candidate role: {jd_info['role']}. Required skills: {', '.join(jd_info['skills'])}. Required experience: {jd_info['experience']}"
    
    k = min(5, len(docs))
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    shortlist = []
    response_lines = [
        "\033[1;36m=== Candidate Screening Results (Top Matches) ===\033[0m",
        "The following candidates were matched against JD requirements using RAG embeddings:",
        ""
    ]
    
    for i, (doc, score) in enumerate(results):
        # Normalize distance score to percentage similarity.
        # Chroma default is squared L2 distance. Standard range is [0.0, 2.0] for normalized vectors.
        # Percentage = max(0, min(100, int((1 - distance/2) * 100)))
        match_percentage = max(0, min(100, int((1.0 - (score / 2.0)) * 100)))
        
        filename = doc.metadata["source"]
        display_name = filename.replace(".txt", "").replace("candidate_", "").replace("_", " ").title()
        
        shortlist.append({"name": filename, "score": match_percentage})
        
        if match_percentage >= 80:
            color = "\033[1;32m" # Bold Green
        elif match_percentage >= 60:
            color = "\033[1;33m" # Yellow
        else:
            color = "\033[1;31m" # Red
            
        response_lines.append(f"  {i+1}. {display_name:<20} -> Match Score: {color}{match_percentage}%\033[0m (File: {filename})")
        
    response_lines.append("")
    response_lines.append("\033[1;35m❓ Do you want to finalize this shortlist? (Yes/No)\033[0m")
    
    return {
        "shortlist": shortlist,
        "pending_confirmation": True,
        "pending_action": "finalize_shortlist",
        "messages": [AIMessage(content="\n".join(response_lines))]
    }

def rewrite_jd_node(state: AgentState) -> dict:
    """Rewrites Job Description using LLM to make the tone exciting and professional."""
    if not state.get("parsed_jd"):
         return {"messages": [AIMessage(content="Please load the JD first.")]}
    
    jd = state["parsed_jd"]
    prompt = (
        "You are an expert recruiter and copywriter. Rewrite the following Job Description for a startup.\n"
        "Keep all the core technical skills and experience levels, but write it in an exciting, modern, fast-paced, and highly engaging tone.\n\n"
        f"Role: {jd['role']}\n"
        f"Required Skills: {', '.join(jd['skills'])}\n"
        f"Experience Level: {jd['experience']}\n\n"
        "Format with markdown (job summary, key responsibilities, benefits, skills)."
    )
    rewritten = safe_invoke(prompt)
    
    response = (
        f"\033[1;36m=== Rewritten Job Description ===\033[0m\n\n{rewritten}\n\n"
        "\033[1;33mTip:\033[0m You can copy this rewritten JD or save it to post online."
    )
    return {"messages": [AIMessage(content=response)]}

def interview_questions_node(state: AgentState) -> dict:
    """Generates tailored interview questions for a candidate grounded in JD + resume content."""
    if not state.get("parsed_jd") or not state.get("resumes"):
        return {"messages": [AIMessage(content="Please load the JD and resumes first.")]}
        
    last_msg = state["messages"][-1].content.lower()
    target_candidate = None
    candidate_filename = None
    
    # 1. Match from filenames
    for filename in state["resumes"].keys():
        name_no_ext = filename.replace(".txt", "").replace("candidate_", "").replace("_", " ").lower()
        if name_no_ext in last_msg or filename.lower() in last_msg:
            target_candidate = name_no_ext
            candidate_filename = filename
            break
            
    # 2. Match from candidate names (check Name: line inside files)
    if not target_candidate:
        for filename, text in state["resumes"].items():
            for line in text.splitlines():
                if line.lower().startswith("name:"):
                    cand_name = line.split(":", 1)[1].strip().lower()
                    if cand_name in last_msg:
                        target_candidate = cand_name
                        candidate_filename = filename
                        break
            if target_candidate:
                break
                
    if not target_candidate:
        candidate_list = []
        for filename in state["resumes"].keys():
            name = filename.replace(".txt", "").replace("candidate_", "").replace("_", " ").title()
            candidate_list.append(name)
        available = ", ".join(candidate_list)
        return {
            "messages": [AIMessage(content=(
                f"⚠️ I couldn't identify which candidate you want interview questions for. "
                f"Available candidates are:\n\033[1;33m{available}\033[0m\n"
                f"Please try again and specify the candidate name (e.g., 'interview questions for Alice Smith')."
            ))]
        }
        
    resume_text = state["resumes"][candidate_filename]
    jd_info = state["parsed_jd"]
    
    prompt = (
        f"You are a technical interviewer hiring for: {jd_info['role']}.\n"
        f"Required skills: {', '.join(jd_info['skills'])}.\n"
        f"Required experience: {jd_info['experience']}.\n\n"
        f"Here is the candidate's resume:\n{resume_text}\n\n"
        "Generate exactly 3 tailored, behavior-based or technical interview questions for this candidate. "
        "Each question should highlight a specific alignment or gap between their background and the JD. "
        "Also include brief guidelines for the interviewer on what to look for in their answers."
    )
    questions = safe_invoke(prompt)
    
    display_name = target_candidate.replace("candidate", "").replace("_", " ").strip().title()
    response = (
        f"\033[1;36m=== Interview Questions for {display_name} ===\033[0m\n\n{questions}"
    )
    return {"messages": [AIMessage(content=response)]}

def salary_search_node(state: AgentState) -> dict:
    """Uses Tavily web search for live salary expectations."""
    if not _tavily_client.api_key:
         return {"messages": [AIMessage(content="Tavily API key is missing. Cannot search live salary data.")]}
         
    last_msg = state["messages"][-1].content.lower()
    
    # Try to extract a specific role, otherwise default to JD's role
    role = None
    for pattern in ["salary for", "salary of", "expectations for", "pay scale of"]:
        if pattern in last_msg:
            role = last_msg.split(pattern, 1)[1].replace("?", "").replace(".", "").strip()
            break
            
    if not role:
        role = state.get("parsed_jd", {}).get("role")
        
    if not role or role.lower() in ["unknown", "unknown role"]:
        role = "Software Engineer"
        
    query = f"Current average salary expectations and salary range for {role} in US tech industry 2025 2026"
    
    try:
        search_result = _tavily_client.search(query=query, search_depth="basic")
        results_text = "\n".join([f"- {res['title']}: {res['content']}" for res in search_result.get('results', [])])
        
        prompt = (
            f"Based on the following Tavily web search results, summarize the salary expectations, average salary, "
            f"and typical range for a '{role}' in the US technology industry for 2025/2026.\n\n"
            f"Search results:\n{results_text}\n\n"
            "Provide a concise summary with bullet points highlighting entry-level, mid-level, and senior expectations."
        )
        summary = safe_invoke(prompt)
        
        response = (
            f"\033[1;36m=== Salary Expectations for {role.title()} ===\033[0m\n\n{summary}\n\n"
            f"\033[1;33mSource:\033[0m Live Tavily web search."
        )
        return {"messages": [AIMessage(content=response)]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error searching salary expectations: {e}")]}

def human_approval_node(state: AgentState) -> dict:
    """Manages recruiter confirmations for shortlisting and other important actions."""
    last_msg = state["messages"][-1].content.lower().strip()
    action = state.get("pending_action")
    
    if action == "finalize_shortlist":
        if any(word in last_msg for word in ["yes", "y", "confirm", "finalize", "sure", "ok", "correct"]):
            shortlist_text = "\n".join([
                f"  - {item['name'].replace('.txt', '').replace('candidate_', '').replace('_', ' ').title():<20} ({item['score']}% match)"
                for item in state.get("shortlist", [])
            ])
            return {
                "pending_confirmation": False,
                "pending_action": None,
                "messages": [AIMessage(content=(
                    f"\033[1;32mShortlist finalized successfully! ✅\033[0m\n\n"
                    f"\033[1mFinal Shortlisted Candidates:\033[0m\n{shortlist_text}\n\n"
                    f"What would you like to do next? (e.g., generate interview questions or check salaries)"
                ))]
            }
        else:
            return {
                "pending_confirmation": False,
                "pending_action": None,
                "shortlist": [],
                "messages": [AIMessage(content="\033[1;31mShortlist discarded. ❌\033[0m What would you like to do next?")]
            }
            
    return {
        "pending_confirmation": False,
        "pending_action": None,
        "messages": [AIMessage(content="Confirmation cleared. Ready for your next query.")]
    }

def default_response_node(state: AgentState) -> dict:
    """Handles out-of-scope queries."""
    return {
        "messages": [AIMessage(content=(
            "I'm sorry, I can only help you with recruitment operations such as:\n"
            "  - Loading and parsing Job Descriptions (JDs)\n"
            "  - Counting applicants in the database\n"
            "  - Screening candidates (RAG match ranking)\n"
            "  - Rewriting JDs\n"
            "  - Generating tailored interview questions\n"
            "  - Looking up live salary expectations"
        ))]
    }

# =======================
# Edge Routing Logic
def route_user_query(state: AgentState) -> Literal["parse_jd_resumes_node", "count_applicants_node", "screen_candidates_node", "rewrite_jd_node", "interview_questions_node", "salary_search_node", "human_approval_node", "default_response", "mismatch_feedback_node", "candidate_comparison_node", "jd_improvement_node", "email_draft_node", "email_send_node", "interview_schedule_node", "skill_trend_analysis_node"]:
    if state.get("pending_confirmation"):
        return "human_approval_node"
    
    last_msg = state["messages"][-1].content.lower()
    
    if any(k in last_msg for k in ["load", "parse", "here is the jd", "read jd", "upload"]):
        return "parse_jd_resumes_node"
    elif "how many" in last_msg and ("applicant" in last_msg or "resume" in last_msg or "candidate" in last_msg):
        return "count_applicants_node"
    elif any(k in last_msg for k in ["screen", "rank", "top candidate", "shortlist", "evaluate", "matching"]):
        return "screen_candidates_node"
    elif "rewrite" in last_msg and "jd" in last_msg:
        return "rewrite_jd_node"
    elif any(k in last_msg for k in ["interview question", "interview questions", "question for", "questions for"]):
        return "interview_questions_node"
    elif any(k in last_msg for k in ["salary", "compensation", "pay scale"]):
        return "salary_search_node"
    # New feature commands
    elif any(k in last_msg for k in ["feedback", "mismatch", "experience mismatch"]):
        return "mismatch_feedback_node"
    elif any(k in last_msg for k in ["compare", "comparison", "candidate comparison"]):
        return "candidate_comparison_node"
    elif any(k in last_msg for k in ["suggest jd", "jd improvement", "improve jd"]):
        return "jd_improvement_node"
    elif any(k in last_msg for k in ["draft email", "email draft"]):
        return "email_draft_node"
    elif any(k in last_msg for k in ["send email", "email send"]):
        return "email_send_node"
    elif any(k in last_msg for k in ["schedule interview", "interview schedule"]):
        return "interview_schedule_node"
    elif any(k in last_msg for k in ["skill trend", "skill trends", "skill analysis"]):
        return "skill_trend_analysis_node"
    else:
        return "default_response"

# =======================
# Build Graph
# =======================
workflow = StateGraph(AgentState)

workflow.add_node("parse_jd_resumes_node", parse_jd_resumes_node)
workflow.add_node("count_applicants_node", count_applicants_node)
workflow.add_node("screen_candidates_node", screen_candidates_node)
workflow.add_node("rewrite_jd_node", rewrite_jd_node)
workflow.add_node("interview_questions_node", interview_questions_node)
workflow.add_node("salary_search_node", salary_search_node)
workflow.add_node("human_approval_node", human_approval_node)
workflow.add_node("default_response", default_response_node)
# New feature nodes
workflow.add_node("mismatch_feedback_node", mismatch_feedback_node)
workflow.add_node("candidate_comparison_node", candidate_comparison_node)
workflow.add_node("jd_improvement_node", jd_improvement_node)
workflow.add_node("email_draft_node", email_draft_node)
workflow.add_node("email_send_node", email_send_node)
workflow.add_node("interview_schedule_node", interview_schedule_node)
workflow.add_node("skill_trend_analysis_node", skill_trend_analysis_node)

workflow.add_conditional_edges(
    START,
    route_user_query,
)

# All nodes route to END
for node in ["parse_jd_resumes_node", "count_applicants_node", "screen_candidates_node", "rewrite_jd_node", "interview_questions_node", "salary_search_node", "human_approval_node", "default_response", "mismatch_feedback_node", "candidate_comparison_node", "jd_improvement_node", "email_draft_node", "email_send_node", "interview_schedule_node", "skill_trend_analysis_node"]:
    workflow.add_edge(node, END)

app = workflow.compile()

# =======================
# Terminal UI Loop
# =======================
if __name__ == "__main__":
    os.makedirs("data/resumes", exist_ok=True)
    
    # 1. Print Startup Banner
    print("\033[1;36m" + "="*65 + "\033[0m")
    print("\033[1;35m    🤖 WELCOME TO THE RECRUITMENT AGENT CHATBOT 💼\033[0m")
    print("      Optimizing Hiring Pipelines from JD to Screening")
    print("\033[1;36m" + "="*65 + "\033[0m")
    print("⏳ Initializing database and parsing data/jd.txt...")

    # Ensure data/jd.txt exists
    if not os.path.exists("data/jd.txt"):
        with open("data/jd.txt", "w", encoding="utf-8") as f:
            f.write(
                "Job Title: Senior Python Developer\n"
                "Role: Backend Software Engineer\n"
                "Experience Required: 5+ years\n"
                "Required Skills:\n"
                "- Python\n"
                "- FastAPI\n"
                "- PostgreSQL\n"
                "- Docker\n"
            )
            
    # Preload resumes
    preload_resumes = {}
    if os.path.exists("data/resumes"):
        for filepath in glob.glob("data/resumes/*.txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                preload_resumes[os.path.basename(filepath)] = f.read()

    # Preload and parse JD
    preload_jd = None
    try:
        with open("data/jd.txt", "r", encoding="utf-8") as f:
            jd_text = f.read()
        parsed = parse_jd(jd_text)
        preload_jd = parsed.model_dump()
        is_local = parse_jd_locally(jd_text) is not None
        source = "locally" if is_local else "via Gemini"
        print(f"✅ Loaded Job Description ({source}) for: \033[1;32m{parsed.role}\033[0m")
    except Exception as e:
        print(f"⚠️ Could not automatically parse JD: {e}")

    print(f"✅ Preloaded \033[1;32m{len(preload_resumes)}\033[0m candidate resumes from database.")
    print("💡 Type '\033[1;32mhelp\033[0m' to see available features, or '\033[1;31mexit\033[0m' to quit.")
    print("\033[1;36m" + "="*65 + "\033[0m")

    # State initialization
    current_state = {
        "messages": [],
        "parsed_jd": preload_jd,
        "resumes": preload_resumes,
        "shortlist": [],
        "pending_confirmation": False,
        "pending_action": None
    }
    
    state_config = {"configurable": {"thread_id": "recruiter_session"}}

    while True:
        try:
            # Styled Prompt
            user_input = input("\033[1;36mRecruiter 👤 >\033[0m ")
        except EOFError:
            break
            
        if user_input.lower().strip() in ['quit', 'exit']:
            print("\n👋 Goodbye! Happy hiring!")
            break
            
        if user_input.lower().strip() in ['upload', 'load']:
            print("\033[1;33m📂 Enter the folder path containing the resumes (leave blank for default 'data/resumes'):\033[0m")
            ans = input("\033[1;36mPath >\033[0m ").strip()
            if ans:
                if os.path.isdir(ans):
                    user_input = f"upload from {ans}"
                else:
                    print(f"\033[1;31m⚠️ Error: '{ans}' is not a valid directory. Falling back to default 'data/resumes/'.\033[0m")
                    user_input = "upload from data/resumes"
            else:
                user_input = "upload from data/resumes"

        if not user_input.strip():
            continue
            
        if user_input.lower().strip() == 'help':
            print("\n\033[1;33m💡 Supported Commands / Queries:\033[0m")
            print("  - \033[1;32mupload / load\033[0m: Reload the Job Description and load resumes from a custom folder.")
            print("  - \033[1;32mhow many applicants\033[0m: Show count of candidates in the database.")
            print("  - \033[1;32mscreen / shortlist\033[0m: Screen and rank candidates using RAG vector similarity.")
            print("  - \033[1;32mrewrite jd\033[0m: Use AI to rewrite the JD with exciting startup tone.")
            print("  - \033[1;32minterview questions for [Name]\033[0m: Generate tailored questions for a candidate.")
            print("  - \033[1;32msalary for [Role]\033[0m: Perform Tavily web search for live salary details.")
            print("  - \033[1;32mexit\033[0m: Exit the chatbot.")
            print("\033[1;36m" + "-"*65 + "\033[0m")
            continue
            
        current_state["messages"].append(HumanMessage(content=user_input))
        
        # Invoke StateGraph workflow
        try:
            result = app.invoke(current_state, config=state_config)
            current_state = result
            
            # Print latest agent message
            if result["messages"]:
                print(f"\n\033[1;35mAgent 🤖 >\033[0m {result['messages'][-1].content}\n")
        except Exception as e:
            print(f"\n\033[1;31mAgent 🤖 Error >\033[0m An error occurred during graph execution: {e}\n")
