import asyncio
import os
import sys

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath("."))

from backend.confidence_rag import (
    handle_query as confidence_rag_handle_query,
    is_broad_department_query,
    _get_broad_departments_fact,
)
from backend.main import (
    build_greeting,
    classify_emotion,
    handle_personalized_reengagement,
    _derive_topic_fallback,
    resume_or_create_session,
)
from backend.entity_mapping import detect_entity
import backend.llm as llm_module

async def run_tests():
    print("=" * 60)
    print("RUNNING FINAL VALIDATION TESTS")
    print("=" * 60)

    # ----------------------------------------------------
    # SCENARIO 1: Returning user personalized greeting & re-engagement
    # ----------------------------------------------------
    print("\n--- TEST 1: Returning User Greeting ---")
    weather_phrase = "It's a bright and sunny day at RNSIT."
    greeting = build_greeting("Sneha", is_returning=True, resumed=True, weather_phrase=weather_phrase)
    print(f"Greeting: {greeting}")
    assert "Sneha" in greeting
    assert "bright and sunny" in greeting or "pleasant" in greeting
    assert "How are you doing today?" in greeting

    print("\n--- TEST 1b: User replies with emotion ---")
    # Tone classification
    assert classify_emotion("I'm actually feeling a little stressed") == "stressed"
    assert classify_emotion("I am doing great and very happy!") == "happy"
    assert classify_emotion("Not so good, feeling down today") == "sad"
    assert classify_emotion("I'm feeling alright") == "neutral"

    # Personalized re-engagement with previous questions (LLM available or fallback)
    mock_session = {
        "previous_questions": [
            "What is the highest package in 2025?",
            "Who are the top recruiters?",
            "Tell me about placement record",
        ]
    }
    ans, topic = await handle_personalized_reengagement("I'm actually feeling a little stressed.", mock_session)
    print(f"User: 'I'm actually feeling a little stressed.'")
    print(f"Nova Response: {ans}")
    print(f"Extracted Topic: {topic}")
    assert "placements" in topic.lower() or "placement" in topic.lower()
    # Check no robotic emotion leak
    assert "detected" not in ans.lower()
    assert "classification" not in ans.lower()
    assert "emotional state" not in ans.lower()
    assert "previous topic was rnsit" not in ans.lower()
    assert "continue from where we left off" not in ans.lower()
    print("TEST 1 PASSED.")

    # ----------------------------------------------------
    # SCENARIO 2: "Which department is good?" (Broad question)
    # ----------------------------------------------------
    print("\n--- TEST 2: Broad Question: 'Which department is good?' ---")
    q2 = "Which department is good?"
    assert is_broad_department_query(q2) is True
    res2 = await confidence_rag_handle_query(q2)
    print(f"Query: {q2}")
    print(f"Route: {res2.get('route')}")
    print(f"Answer: {res2.get('answer')}")
    assert "i don't have that detail" not in res2["answer"].lower()
    assert "i don't know" not in res2["answer"].lower()
    # Should mention software/computing or CSE/ISE or electronics/departments
    assert any(w in res2["answer"].lower() for w in ("cse", "ise", "software", "department", "departments", "engineering", "electronics"))
    print("TEST 2 PASSED.")

    # ----------------------------------------------------
    # SCENARIO 3: "Tell me about CSE." (Specific Entity)
    # ----------------------------------------------------
    print("\n--- TEST 3: Specific Question: 'Tell me about CSE.' ---")
    q3 = "Tell me about CSE."
    detected3 = detect_entity(q3)
    print(f"Query: {q3}")
    print(f"Detected Entity: {detected3.entity_id if detected3 else None}")
    assert detected3 is not None
    assert detected3.entity_id == "CSE_DEPT"
    res3 = await confidence_rag_handle_query(q3)
    print(f"Route: {res3.get('route')}")
    print(f"Answer: {res3.get('answer')}")
    assert "kiran" in res3["answer"].lower() or "720" in res3["answer"].lower() or "computer science" in res3["answer"].lower()
    print("TEST 3 PASSED.")

    # ----------------------------------------------------
    # SCENARIO 4: "What is the placement information?"
    # ----------------------------------------------------
    print("\n--- TEST 4: Placement Question: 'What is the placement information?' ---")
    q4 = "What is the placement information?"
    detected4 = detect_entity(q4)
    print(f"Query: {q4}")
    print(f"Detected Entity: {detected4.entity_id if detected4 else None}")
    assert detected4 is not None
    assert detected4.entity_id == "PLACEMENTS_OVERVIEW"
    res4 = await confidence_rag_handle_query(q4)
    print(f"Route: {res4.get('route')}")
    print(f"Answer: {res4.get('answer')}")
    assert "200" in res4["answer"] or "recruit" in res4["answer"].lower() or "placement" in res4["answer"].lower()
    print("TEST 4 PASSED.")

    # ----------------------------------------------------
    # SCENARIO 5: Unsupported / non-RNSIT question (Guardrail)
    # ----------------------------------------------------
    print("\n--- TEST 5: Unsupported Question: 'Who is the president of France?' ---")
    q5 = "Who is the president of France?"
    res5 = await confidence_rag_handle_query(q5)
    print(f"Query: {q5}")
    print(f"Route: {res5.get('route')}")
    print(f"Answer: {res5.get('answer')}")
    assert res5["route"] == "LOW_OUT_OF_SCOPE" or "rnsit" in res5["answer"].lower()
    print("TEST 5 PASSED.")

    # ----------------------------------------------------
    # SCENARIO 6: LLM unavailable deterministic fallback
    # ----------------------------------------------------
    print("\n--- TEST 6: Deterministic Fallback when LLM is unavailable ---")
    # Temporarily monkeypatch chat_completion_with_fallback to fail
    orig_fallback = llm_module.chat_completion_with_fallback
    async def mock_fail(*args, **kwargs):
        raise RuntimeError("Simulated LLM outage")
    llm_module.chat_completion_with_fallback = mock_fail

    try:
        # 6a: Re-engagement fallback
        ans6, topic6 = await handle_personalized_reengagement("I'm actually feeling a little stressed.", mock_session)
        print(f"Fallback Re-engagement: {ans6}")
        assert "I'm sorry to hear that" in ans6 or "Take it easy" in ans6
        assert "placements" in ans6.lower()
        assert "Would you like to continue" in ans6

        # 6b: Broad department fallback
        res6b = await confidence_rag_handle_query("Which department is good?")
        print(f"Fallback Broad Dept: {res6b.get('answer')}")
        assert "i don't have that detail" not in res6b["answer"].lower()
        assert "RNSIT" in res6b["answer"] or "department" in res6b["answer"].lower()

        # 6c: Specific entity fallback
        res6c = await confidence_rag_handle_query("Tell me about CSE.")
        print(f"Fallback Specific Entity: {res6c.get('answer')}")
        assert "kiran" in res6c["answer"].lower() or "720" in res6c["answer"] or "cse" in res6c["answer"].lower()
        print("TEST 6 PASSED.")
    finally:
        llm_module.chat_completion_with_fallback = orig_fallback

    print("\n" + "=" * 60)
    print("ALL 6 TEST SCENARIOS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_tests())
