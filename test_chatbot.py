import os
import sys
import glob
from langchain_core.messages import HumanMessage, AIMessage

# Reconfigure stdout/stderr to UTF-8 on Windows to support emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from chatbot import app, ParsedJD

def test_recruitment_agent():
    print("==================================================")
    print("Running Recruitment Agent Integration Tests...")
    print("==================================================")

    # 1. Load mock data for test state
    resumes = {}
    if os.path.exists("data/resumes"):
        for filepath in glob.glob("data/resumes/*.txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                resumes[os.path.basename(filepath)] = f.read()

    test_jd = {
        "role": "Senior Python Developer",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "experience": "5+ years"
    }

    # Initial state
    state = {
        "messages": [],
        "parsed_jd": test_jd,
        "resumes": resumes,
        "shortlist": [],
        "pending_confirmation": False,
        "pending_action": None
    }

    state_config = {"configurable": {"thread_id": "test_session"}}

    # Helper function to run a turn and print response
    def run_turn(query: str):
        import time
        time.sleep(2.0)
        nonlocal state
        print(f"\nRecruiter: {query}")
        state["messages"].append(HumanMessage(content=query))
        result = app.invoke(state, config=state_config)
        state = result
        last_msg = result["messages"][-1].content
        # Strip ANSI colors for test output print
        clean_msg = last_msg.replace("\033[1;36m", "").replace("\033[1;35m", "").replace("\033[1;32m", "").replace("\033[1;33m", "").replace("\033[1;31m", "").replace("\033[0m", "").replace("\033[1", "")
        print(f"Agent: {clean_msg}")
        return result

    # Test 1: Count applicants (Plain Python)
    print("\n--- Test 1: Applicant Count ---")
    res = run_turn("How many applicants are there?")
    assert len(res["messages"]) > 0

    # Test 2: RAG-based screening
    print("\n--- Test 2: Candidate Screening ---")
    res = run_turn("Screen candidates")
    assert res.get("pending_confirmation") is True
    assert res.get("pending_action") == "finalize_shortlist"
    assert len(res.get("shortlist", [])) > 0

    # Test 3: Confirmation loop (Yes)
    print("\n--- Test 3: Confirm Shortlist ---")
    res = run_turn("Yes, finalize the shortlist")
    assert res.get("pending_confirmation") is False
    assert res.get("pending_action") is None

    # Test 4: Interview Questions Generation
    print("\n--- Test 4: Interview Questions ---")
    # Grace Hopper is one of the candidates generated in data/resumes
    res = run_turn("Generate interview questions for Grace Hopper")
    assert len(res["messages"]) > 0

    # Test 5: JD Rewriting
    print("\n--- Test 5: JD Rewriting ---")
    res = run_turn("Rewrite the jd")
    assert len(res["messages"]) > 0

    # Test 6: Salary expectations search
    print("\n--- Test 6: Salary Search ---")
    res = run_turn("What is the salary expectations for a Python Developer?")
    assert len(res["messages"]) > 0

    # Test 7: Load/upload resumes from custom directory
    print("\n--- Test 7: Custom Directory Upload ---")
    res = run_turn("upload from data/resumes")
    assert len(res["resumes"]) > 0
    assert "resumes" in res["messages"][-1].content.lower()

    print("\n==================================================")
    print("All Integration Tests Passed Successfully!")
    print("==================================================")

if __name__ == "__main__":
    test_recruitment_agent()
