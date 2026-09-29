"""
backend/entity_mapping.py — Reliable Canonical Question-to-Entity Mapping
========================================================================
Maps user questions to canonical college entities defined in `college_info.json`.
Handles varied phrasings, aliases, synonyms, and STT transcriptions to ensure:
  - "Where is the library?", "Where can I find the library?", "Library location?" -> ENTITY: LIBRARY
  - "Who is the HOD of CSE?", "Who heads the computer science department?", "CSE department HOD?" -> ENTITY: CSE_HOD
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import NamedTuple

from backend.query_correction import normalize_query

logger = logging.getLogger("RNSIT_Kiosk.EntityMapping")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "college_info.json")


class DetectedEntity(NamedTuple):
    entity_id: str
    canonical_name: str
    verified_answer: str
    confidence: float
    entity_type: str


# Load college_info.json once at module load
_KB_DATA: dict = {}
if os.path.exists(DATA_PATH):
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _KB_DATA = json.load(f)
    except Exception as err:
        logger.error(f"[EntityMapping] Failed to load college_info.json: {err}")


def get_kb_data() -> dict:
    return _KB_DATA


# ── Canonical Entities & Verified Answers ─────────────────────────────────
# Extracted directly from college_info.json to ensure 100% fidelity.

def _build_verified_answers() -> dict[str, dict]:
    kb = _KB_DATA
    college = kb.get("college", {})
    admin = kb.get("administration", {})
    depts = kb.get("departments", {})
    facs = kb.get("facilities", {})
    placements = kb.get("placements", {})
    admissions = kb.get("admissions", {})
    faqs = kb.get("faqs", {})

    faq_map = {f["question"].strip().lower(): f["answer"].strip() for f in faqs if "question" in f and "answer" in f}

    cse = depts.get("cse", {})
    aiml = depts.get("aiml", {})
    lib = facs.get("library", {})
    canteen = facs.get("canteen", {})
    aud = facs.get("auditorium", {})
    gym = facs.get("gym", {})
    b_hostel = facs.get("hostels", {}).get("boys", {})
    g_hostel = facs.get("hostels", {}).get("girls", {})
    sports = facs.get("sports", {})
    placement_cell = facs.get("placement_cell", {})
    medical = facs.get("medical", {})
    wifi = facs.get("wifi", {})
    banking = facs.get("banking", {})
    toyota = facs.get("toyota_center", {})
    counselling = facs.get("counselling", {})

    stats_2025 = placements.get("stats", {}).get("2025", {})
    stats_2023 = placements.get("stats", {}).get("2023", {})

    return {
        "PRINCIPAL": {
            "name": "Principal",
            "type": "administration",
            "answer": faq_map.get(
                "who is the principal of rnsit?",
                f"Dr. Ramesh Babu H S is the current Principal of RNSIT. Contact: {admin.get('principal', {}).get('phone', '+91 94489 53800')}."
            ),
        },
        "DIRECTOR": {
            "name": "Director",
            "type": "administration",
            "answer": faq_map.get(
                "who is the director of rnsit?",
                f"Dr. M K Venkatesha is the Director of RNSIT. Contact: {admin.get('director', {}).get('phone', '+91 98808 67337')}."
            ),
        },
        "CHAIRMAN_FOUNDER": {
            "name": "Group Chairman & Founder",
            "type": "administration",
            "answer": faq_map.get(
                "who is the chairman of rnsit?",
                f"Dr. R. N. Shetty is the Group Chairman and founder of RNSIT."
            ),
        },
        "ESTABLISHED_YEAR": {
            "name": "Established Year",
            "type": "college",
            "answer": faq_map.get(
                "when was rnsit established?",
                f"RNSIT was established in the year {college.get('established', 2001)}."
            ),
        },
        "COLLEGE_OVERVIEW": {
            "name": "College Overview",
            "type": "college",
            "answer": (
                f"RNS Institute of Technology (RNSIT) was established in 2001 by Dr. R. N. Shetty. "
                f"It is an autonomous private engineering college affiliated to VTU, located in Channasandra, Bengaluru."
            ),
        },
        "AFFILIATION": {
            "name": "Affiliation & Autonomy",
            "type": "college",
            "answer": faq_map.get(
                "what is the affiliation of rnsit?",
                "RNSIT is affiliated to Visvesvaraya Technological University (VTU) and is an Autonomous institution."
            ),
        },
        "COLLEGE_LOCATION": {
            "name": "College Location",
            "type": "college",
            "answer": faq_map.get(
                "where is rnsit located?",
                f"RNSIT is located at {college.get('location', 'R R Nagar Post, Channasandra, Bengaluru - 560098')}."
            ),
        },
        "WEBSITE": {
            "name": "Official Website",
            "type": "college",
            "answer": faq_map.get(
                "what is the website of rnsit?",
                f"The official website of RNSIT is {college.get('website', 'https://www.rnsit.ac.in')}."
            ),
        },
        "WORKING_HOURS": {
            "name": "Working Hours & Timings",
            "type": "administration",
            "answer": faq_map.get(
                "what are the college working hours?",
                f"The college working hours are {admin.get('working_hours', '9:20 AM to 5:00 PM on all working days')}."
            ),
        },
        "ADMIN_BLOCK": {
            "name": "Administrative Block",
            "type": "facility",
            "answer": faq_map.get(
                "where is admin block?",
                "The Administrative Block is located opposite the Central Library and houses the Director and Principal offices, Accounts, Admissions, Placement Cell, and the Dr. Shivaram Karanth Auditorium."
            ),
        },
        "ACCOUNTS_OFFICE": {
            "name": "Accounts Office",
            "type": "administration",
            "answer": faq_map.get(
                "where is the accounts manager office?",
                "The Accounts office is located in the Administrative Block."
            ),
        },
        "ADMISSIONS_CONTACT": {
            "name": "Admissions Contact",
            "type": "admissions",
            "answer": faq_map.get(
                "how can i contact the admissions department?",
                f"Call {admin.get('contacts', {}).get('admissions_phone', '+91 81472 86667')} or email {admin.get('contacts', {}).get('admissions_email', 'admissions@rnsit.ac.in')}. The admissions office is open on all Saturdays and Sundays too."
            ),
        },
        "GENERAL_ENQUIRY": {
            "name": "General Enquiry",
            "type": "administration",
            "answer": (
                f"RNSIT general enquiry phone numbers are: {', '.join(admin.get('contacts', {}).get('enquiry_phone', ['+91 80286 11880', '+91 80286 11881', '+91 80286 11882']))}."
            ),
        },
        "COMEDK_CODE": {
            "name": "COMEDK Code",
            "type": "admissions",
            "answer": faq_map.get(
                "what is the comedk code for rnsit?",
                f"The COMEDK code for RNSIT is {admissions.get('comedk_code', college.get('admission_codes', {}).get('comedk', 'E104'))}."
            ),
        },
        "CET_CODE": {
            "name": "CET Code",
            "type": "admissions",
            "answer": faq_map.get(
                "what is the cet code for rnsit?",
                f"The CET code for RNSIT is {admissions.get('cet_code', college.get('admission_codes', {}).get('cet', 'E118'))}."
            ),
        },
        "ADMISSION_CODES": {
            "name": "Admission Codes",
            "type": "admissions",
            "answer": (
                f"The CET code for RNSIT is {college.get('admission_codes', {}).get('cet', 'E118')} "
                f"and the COMEDK code is {college.get('admission_codes', {}).get('comedk', 'E104')}."
            ),
        },
        "ADMISSIONS_PROCESS": {
            "name": "Admission Process & Modes",
            "type": "admissions",
            "answer": (
                "RNSIT admissions for B.E. are conducted via KCET, COMEDK UGET, JEE Main, and Management Quota. "
                "For postgraduate programs, entry is via Karnataka PGCET, KMAT, or GATE. Candidates must complete counseling, document verification, and fee payment."
            ),
        },
        "ADMISSIONS_ELIGIBILITY": {
            "name": "Admission Eligibility",
            "type": "admissions",
            "answer": admissions.get("eligibility", {}).get(
                "be",
                "Pass in 10+2 / PUC with Physics and Mathematics compulsory, plus Chemistry/Biology/Electronics/CS, and a valid KCET, COMEDK, or JEE Main score."
            ),
        },
        "ADMISSIONS_DOCUMENTS": {
            "name": "Documents Required",
            "type": "admissions",
            "answer": (
                "Documents required for RNSIT admission: " +
                "; ".join(admissions.get("documents_required", [
                    "10th & 12th marks cards", "KCET/COMEDK/JEE rank card", "Transfer & migration certificates", "Photos", "Aadhaar card"
                ])) + "."
            ),
        },
        "FEES": {
            "name": "Fee Structure",
            "type": "admissions",
            "answer": (
                "B.E. annual fees at RNSIT vary significantly by quota (Government KCET vs COMEDK / Management) "
                "and branch, typically ranging from roughly 1.5 to 7.5 lakh rupees per year. Please verify the exact current year fee with the Admissions Office at +91 81472 86667."
            ),
        },
        "MANAGEMENT_FEES": {
            "name": "Management Quota Fees",
            "type": "admissions",
            "answer": faq_map.get(
                "what is the management fee of rnsit?",
                "Management quota annual fees for B.E. programs at RNSIT typically range from 2.5 to 7.5 lakh rupees per year depending on the branch. For exact current year fee details, contact the Admissions Office at +91 81472 86667."
            ),
        },
        "SCHOLARSHIPS": {
            "name": "Scholarships",
            "type": "admissions",
            "answer": (
                "RNSIT supports Karnataka government scholarships for eligible SC/ST/OBC/minority categories, "
                "National Scholarship Portal (NSP) schemes, and merit-based fee concessions. Please verify eligibility with the Admissions Office."
            ),
        },
        "DEPARTMENTS_OVERVIEW": {
            "name": "Departments Overview",
            "type": "department",
            "answer": faq_map.get(
                "what departments does rnsit have?",
                "RNSIT offers CSE, ECE, ISE, EEE, Mechanical, Civil, AI and ML, CSE Data Science, CSE Cyber Security, AI and DS at UG level, and MBA and MCA at PG level."
            ),
        },
        "CSE_HOD": {
            "name": "HOD of Computer Science and Engineering",
            "type": "department",
            "answer": faq_map.get(
                "who is the hod of cse?",
                f"Dr. Kiran Y.C. is the HOD of the CSE department at RNSIT."
            ),
        },
        "CSE_INTAKE": {
            "name": "CSE Department Intake",
            "type": "department",
            "answer": faq_map.get(
                "what is the intake for the cse department?",
                f"The CSE department has an intake of {cse.get('intake', 720)} students per year."
            ),
        },
        "CSE_DEPT": {
            "name": "Computer Science and Engineering Department",
            "type": "department",
            "answer": (
                f"The Department of Computer Science and Engineering is located in the {cse.get('block', 'CSE Block')}. "
                f"The HOD is {cse.get('hod', 'Dr. Kiran Y.C.')} and the annual intake is {cse.get('intake', 720)} students. It also has a PhD research center."
            ),
        },
        "AIML_HOD": {
            "name": "HOD of AI and ML",
            "type": "department",
            "answer": faq_map.get(
                "who is the hod of ai ml?",
                f"Dr. Andhe Pallavi is the HOD of the AI and ML department at RNSIT."
            ),
        },
        "AIML_DEPT": {
            "name": "CSE (AI and ML) Department",
            "type": "department",
            "answer": (
                f"The Department of CSE (AI and ML) is located in the {aiml.get('block', 'CSE Block')}. "
                f"The HOD is {aiml.get('hod', 'Dr. Andhe Pallavi')}."
            ),
        },
        "ECE_DEPT": {
            "name": "Electronics and Communication Engineering",
            "type": "department",
            "answer": "The Electronics and Communication Engineering (ECE) department is located in the Main Campus and features a PhD research center.",
        },
        "ISE_DEPT": {
            "name": "Information Science and Engineering",
            "type": "department",
            "answer": "The Information Science and Engineering (ISE) department is located in the CSE Block.",
        },
        "EEE_DEPT": {
            "name": "Electrical and Electronics Engineering",
            "type": "department",
            "answer": "RNSIT offers Electrical and Electronics Engineering (EEE) with state-of-the-art laboratory facilities.",
        },
        "MECH_DEPT": {
            "name": "Mechanical Engineering",
            "type": "department",
            "answer": "The Mechanical Engineering department is located in the Mechanical Block, houses the Toyota Center of Excellence, and has a PhD research center.",
        },
        "CIVIL_DEPT": {
            "name": "Civil Engineering",
            "type": "department",
            "answer": "The Civil Engineering department is located in the Civil Block and includes a recognized PhD research center.",
        },
        "CSE_DS_DEPT": {
            "name": "CSE (Data Science)",
            "type": "department",
            "answer": "The CSE (Data Science) department is located in the CSE Block.",
        },
        "CSE_CS_DEPT": {
            "name": "CSE (Cyber Security)",
            "type": "department",
            "answer": "The CSE (Cyber Security) department is located in the CSE Block.",
        },
        "AIDS_DEPT": {
            "name": "Artificial Intelligence and Data Science",
            "type": "department",
            "answer": "RNSIT offers Artificial Intelligence and Data Science (AI & DS) as an undergraduate engineering specialization.",
        },
        "MCA_DEPT": {
            "name": "Master of Computer Applications",
            "type": "department",
            "answer": "The Master of Computer Applications (MCA) department is located in the MBA Block.",
        },
        "MBA_DEPT": {
            "name": "Master of Business Administration",
            "type": "department",
            "answer": "The Master of Business Administration (MBA) department is located in the MBA Block and offers a PhD research center.",
        },
        "PHD_PROGRAMS": {
            "name": "PhD Programs",
            "type": "academics",
            "answer": faq_map.get(
                "does rnsit have phd programs?",
                "Yes, RNSIT has VTU Research Centers in CSE, ECE, Mechanical, Civil, Chemistry, Physics, Math, and MBA."
            ),
        },
        "LIBRARY": {
            "name": "Central Library",
            "type": "facility",
            "answer": (
                f"The RNSIT Central Library is located {lib.get('location', 'Opposite the Administrative Block')} "
                f"and is open from {lib.get('timings', '09:00 AM to 08:00 PM')}."
            ),
        },
        "LIBRARY_TIMINGS": {
            "name": "Library Timings",
            "type": "facility",
            "answer": faq_map.get(
                "what are the library timings?",
                f"The RNSIT Central Library is open from {lib.get('timings', '09:00 AM to 08:00 PM')}."
            ),
        },
        "CANTEEN": {
            "name": "Canteen",
            "type": "facility",
            "answer": faq_map.get(
                "where is the canteen located?",
                f"The canteen is located {canteen.get('location', 'near the sports ground and hostel entrance')} on the RNSIT campus."
            ),
        },
        "AUDITORIUM": {
            "name": "Dr. Shivaram Karanth Auditorium",
            "type": "facility",
            "answer": faq_map.get(
                "where is the dr. shivaram karanth auditorium?",
                f"The Dr. Shivaram Karanth Auditorium is located in the {aud.get('location', 'Administrative Block')} with a seating capacity of {aud.get('seating_capacity', 1800)}."
            ),
        },
        "GYM": {
            "name": "Gymnasium",
            "type": "facility",
            "answer": faq_map.get(
                "is there a gym on campus?",
                f"Yes, RNSIT has {gym.get('details', 'separate gym facilities and timings for boys and girls')}."
            ),
        },
        "HOSTELS": {
            "name": "Hostels",
            "type": "facility",
            "answer": (
                f"RNSIT provides on-campus hostels. The boys hostel accommodates {b_hostel.get('capacity', 530)} students on a shared basis, "
                f"and the girls hostel accommodates {g_hostel.get('capacity', 300)} students with amenities including yoga and meditation."
            ),
        },
        "BOYS_HOSTEL": {
            "name": "Boys Hostel",
            "type": "facility",
            "answer": f"The boys hostel has a capacity of {b_hostel.get('capacity', 530)} students on a shared basis.",
        },
        "GIRLS_HOSTEL": {
            "name": "Girls Hostel",
            "type": "facility",
            "answer": faq_map.get(
                "how many students can the girls hostel accommodate?",
                f"The girls hostel can accommodate {g_hostel.get('capacity', 300)} students and offers amenities like Morning Yoga and Meditation sessions."
            ),
        },
        "SPORTS": {
            "name": "Sports Ground & Courts",
            "type": "facility",
            "answer": (
                f"RNSIT features a {sports.get('cricket_ground_sqm', 17000)} square-meter cricket ground "
                f"as well as {', '.join(sports.get('courts', ['Basketball', 'Football']))} courts."
            ),
        },
        "PLACEMENT_CELL": {
            "name": "Placement Cell",
            "type": "facility",
            "answer": faq_map.get(
                "where is the placement cell?",
                f"The placement cell is located on the {placement_cell.get('location', 'ground floor of the Admin Block')}."
            ),
        },
        "PLACEMENTS_OVERVIEW": {
            "name": "Placements Overview",
            "type": "placements",
            "answer": faq_map.get(
                "how many companies recruit from rnsit?",
                f"More than {placements.get('total_companies', '200+')} companies recruit from RNSIT including Cognizant, Infosys, IBM, HCL, Tata Elxsi, and many more."
            ),
        },
        "PLACEMENTS_HIGHEST": {
            "name": "Highest Placement Package",
            "type": "placements",
            "answer": faq_map.get(
                "what is the highest placement package offered?",
                f"The highest CTC offered was {stats_2023.get('highest_ctc_lpa', 56)} LPA in 2023. In 2025, it was {stats_2025.get('highest_ctc_lpa', 50)} LPA."
            ),
        },
        "PLACEMENTS_2025": {
            "name": "Placement Stats 2025",
            "type": "placements",
            "answer": faq_map.get(
                "what is the highest package in 2025?",
                f"In 2025, RNSIT achieved {stats_2025.get('placements', 1060)} placements with a highest CTC of {stats_2025.get('highest_ctc_lpa', 50)} LPA and {stats_2025.get('internships', 426)} internships."
            ),
        },
        "MEDICAL": {
            "name": "Medical & Ambulance",
            "type": "facility",
            "answer": faq_map.get(
                "is medical help available on campus?",
                f"Yes, {medical.get('details', 'first-aid kits and 24/7 ambulance services are available on campus')}."
            ),
        },
        "WIFI": {
            "name": "Wi-Fi Facility",
            "type": "facility",
            "answer": faq_map.get(
                "does rnsit have wi-fi?",
                f"Yes, {wifi.get('details', 'RNSIT has campus-wide Wi-Fi available for all students and staff')}."
            ),
        },
        "BANKING_ATM": {
            "name": "Banking Counter & ATM",
            "type": "facility",
            "answer": faq_map.get(
                "is there an atm on campus?",
                f"Yes, {banking.get('details', 'a Canara Bank extension counter and ATM are available on the RNSIT campus')}."
            ),
        },
        "TOYOTA_CENTER": {
            "name": "Toyota Center of Excellence",
            "type": "facility",
            "answer": faq_map.get(
                "where can i find the toyota center of excellence?",
                f"The Toyota Center of Excellence is located in the {toyota.get('location', 'Mechanical Engineering department')}."
            ),
        },
        "TRANSPORT": {
            "name": "Transport Facility",
            "type": "facility",
            "answer": faq_map.get(
                "does the college provide transport?",
                "Yes, RNSIT operates a fleet of buses and a van for students."
            ),
        },
        "COUNSELLING": {
            "name": "Student Counselling",
            "type": "facility",
            "answer": (
                f"One-on-one sessions with college counsellors are available for students. "
                f"Bookings can be made at {counselling.get('booking_url', 'https://calendly.com/rnsit-website/15min')}."
            ),
        },
        "CLUBS": {
            "name": "Student Clubs",
            "type": "academics",
            "answer": faq_map.get(
                "what are the student clubs at rnsit?",
                f"RNSIT has clubs including {', '.join(kb.get('academics', {}).get('clubs', ['QuizCorp', 'Google DSC', 'AdroIT', 'Big O Coding Club', 'Placement Club', 'Aura']))}."
            ),
        },
        "CULTURAL_FEST": {
            "name": "Cultural Fest",
            "type": "academics",
            "answer": faq_map.get(
                "what is the cultural fest of rnsit?",
                f"The cultural fest of RNSIT is called {kb.get('academics', {}).get('cultural_fest', 'Aperture')}."
            ),
        },
        "CAPABILITIES": {
            "name": "Nova Capabilities",
            "type": "meta",
            "answer": faq_map.get(
                "what can you do",
                "I can help you with information about RNSIT departments, facilities, staff, admissions, placements, library, hostel, canteen, and much more. Just ask me anything!"
            ),
        },
    }


VERIFIED_ENTITIES = _build_verified_answers()


# ── Pattern & Keyword Matchers ───────────────────────────────────────────
# Maps user expressions to canonical entity IDs.
# Order matters: more specific entities (e.g. CSE_HOD, LIBRARY_TIMINGS)
# precede broader entities (CSE_DEPT, LIBRARY).

_ENTITY_RULES: list[tuple[str, list[re.Pattern]]] = [
    # 1. HODs
    (
        "CSE_HOD",
        [
            re.compile(r"\b(?:hod|head)\s+(?:of\s+)?(?:the\s+)?(?:cse|computer\s+science)\b"),
            re.compile(r"\b(?:cse|computer\s+science)\s+(?:department\s+)?(?:hod|head)\b"),
            re.compile(r"\bwho\s+heads\s+(?:the\s+)?(?:cse|computer\s+science)\b"),
            re.compile(r"\bdr\.?\s*kiran\s*(?:y\.?\s*c\.?)?\b"),
            re.compile(r"\bkiran\s+y\s*c\b"),
        ],
    ),
    (
        "AIML_HOD",
        [
            re.compile(r"\b(?:hod|head)\s+(?:of\s+)?(?:the\s+)?(?:ai\s*ml|aiml|ai\s+and\s+ml|artificial\s+intelligence)\b"),
            re.compile(r"\b(?:ai\s*ml|aiml|ai\s+and\s+ml)\s+(?:department\s+)?(?:hod|head)\b"),
            re.compile(r"\bwho\s+heads\s+(?:the\s+)?(?:ai\s*ml|aiml)\b"),
            re.compile(r"\bdr\.?\s*andhe\s+pallavi\b"),
            re.compile(r"\bandhe\s+pallavi\b"),
        ],
    ),
    # 2. Key Administration
    (
        "PRINCIPAL",
        [
            re.compile(r"\b(?:who\s+is\s+(?:the\s+)?)?principal\b"),
            re.compile(r"\bdr\.?\s*ramesh\s+babu\b"),
            re.compile(r"\bramesh\s+babu\b"),
            re.compile(r"\bprincipal\s+(?:office|contact|number|phone)\b"),
        ],
    ),
    (
        "DIRECTOR",
        [
            re.compile(r"\b(?:who\s+is\s+(?:the\s+)?)?director\b"),
            re.compile(r"\bdr\.?\s*m\s*k\s*venkatesha\b"),
            re.compile(r"\bm\s*k\s*venkatesha\b"),
            re.compile(r"\bdirector\s+(?:office|contact|number|phone)\b"),
        ],
    ),
    (
        "CHAIRMAN_FOUNDER",
        [
            re.compile(r"\b(?:who\s+is\s+(?:the\s+)?)?(?:group\s+)?chairman\b"),
            re.compile(r"\bwho\s+founded\b"),
            re.compile(r"\bwho\s+is\s+(?:the\s+)?founder\b"),
            re.compile(r"\bdr\.?\s*r\s*n\s*shetty\b"),
            re.compile(r"\br\s*n\s*shetty\b"),
        ],
    ),
    (
        "ESTABLISHED_YEAR",
        [
            re.compile(r"\b(?:when\s+was|year\s+of)\s+rnsit\s+(?:established|founded|started)\b"),
            re.compile(r"\bwhen\s+(?:was\s+)?(?:the\s+)?college\s+(?:established|founded|started)\b"),
            re.compile(r"\bestablished\s+year\b"),
            re.compile(r"\bfounding\s+year\b"),
        ],
    ),
    # 3. Timings & Working Hours
    (
        "LIBRARY_TIMINGS",
        [
            re.compile(r"\blibrary\s+(?:timings?|hours?|schedule)\b"),
            re.compile(r"\bwhen\s+does\s+(?:the\s+)?library\s+(?:open|close)\b"),
            re.compile(r"\btimings?\s+of\s+(?:the\s+)?library\b"),
        ],
    ),
    (
        "WORKING_HOURS",
        [
            re.compile(r"\bcollege\s+(?:working\s+hours|timings?)\b"),
            re.compile(r"\bworking\s+hours\b"),
            re.compile(r"\btimings?\s+of\s+(?:the\s+)?college\b"),
            re.compile(r"\bcollege\s+open(?:ing)?\s+hours?\b"),
            re.compile(r"\bwhen\s+does\s+(?:the\s+)?college\s+(?:open|close)\b"),
        ],
    ),
    # 4. Facilities
    (
        "LIBRARY",
        [
            re.compile(r"\bwhere\s+(?:is|can\s+i\s+find)\s+(?:the\s+)?(?:central\s+)?library\b"),
            re.compile(r"\b(?:central\s+)?library\s+location\b"),
            re.compile(r"\bfind\s+(?:the\s+)?library\b"),
            re.compile(r"\blocation\s+of\s+(?:the\s+)?library\b"),
            re.compile(r"\bwhere\s+to\s+find\s+books\b"),
            re.compile(r"\bk\s*nimbus\b"),
            re.compile(r"\bcentral\s+library\b"),
            re.compile(r"\blibrary\b"),
        ],
    ),
    (
        "CANTEEN",
        [
            re.compile(r"\bwhere\s+(?:is|can\s+i\s+find)\s+(?:the\s+)?canteen\b"),
            re.compile(r"\bcanteen\s+location\b"),
            re.compile(r"\bcanteen\b"),
            re.compile(r"\bcafeteria\b"),
            re.compile(r"\bfood\s+court\b"),
            re.compile(r"\bwhere\s+to\s+eat\b"),
        ],
    ),
    (
        "AUDITORIUM",
        [
            re.compile(r"\b(?:dr\.?\s*)?shivaram\s+karanth\s+auditorium\b"),
            re.compile(r"\bauditorium\b"),
            re.compile(r"\bseminar\s+hall\b"),
        ],
    ),
    (
        "GYM",
        [
            re.compile(r"\b(?:is\s+there\s+a\s+)?gym\b"),
            re.compile(r"\bgymnasium\b"),
            re.compile(r"\bfitness\s+center\b"),
            re.compile(r"\bworkout\b"),
        ],
    ),
    (
        "GIRLS_HOSTEL",
        [
            re.compile(r"\bgirls?\s+hostel\b"),
            re.compile(r"\bhostel\s+for\s+girls?\b"),
            re.compile(r"\bwomen'?s\s+hostel\b"),
            re.compile(r"\bladies\s+hostel\b"),
        ],
    ),
    (
        "BOYS_HOSTEL",
        [
            re.compile(r"\bboys?\s+hostel\b"),
            re.compile(r"\bhostel\s+for\s+boys?\b"),
            re.compile(r"\bmen'?s\s+hostel\b"),
        ],
    ),
    (
        "HOSTELS",
        [
            re.compile(r"\bhostels?\b"),
            re.compile(r"\baccommodation\b"),
            re.compile(r"\bboarding\b"),
        ],
    ),
    (
        "SPORTS",
        [
            re.compile(r"\bsports\s+(?:facilities?|ground)\b"),
            re.compile(r"\bcricket\s+ground\b"),
            re.compile(r"\bbasketball\s+court\b"),
            re.compile(r"\bfootball\s+(?:ground|court)\b"),
            re.compile(r"\bplayground\b"),
        ],
    ),
    (
        "ACCOUNTS_OFFICE",
        [
            re.compile(r"\baccounts?\s+(?:manager|office|section|counter)\b"),
            re.compile(r"\bwhere\s+(?:to|can\s+i)\s+pay\s+(?:the\s+)?fees?\b"),
            re.compile(r"\bfee\s+payment\s+counter\b"),
        ],
    ),
    (
        "ADMIN_BLOCK",
        [
            re.compile(r"\bwhere\s+is\s+(?:the\s+)?admin(?:istrative)?\s+block\b"),
            re.compile(r"\badmin(?:istrative)?\s+block\b"),
            re.compile(r"\badmin\s+building\b"),
        ],
    ),
    (
        "PLACEMENT_CELL",
        [
            re.compile(r"\bwhere\s+is\s+(?:the\s+)?placement\s+cell\b"),
            re.compile(r"\bplacement\s+cell\b"),
            re.compile(r"\btraining\s+and\s+placement\s+office\b"),
            re.compile(r"\btpo\s+office\b"),
        ],
    ),
    (
        "MEDICAL",
        [
            re.compile(r"\bmedical\s+(?:help|care|facility|room)\b"),
            re.compile(r"\bfirst\s*aid\b"),
            re.compile(r"\bambulance\b"),
            re.compile(r"\bdoctor\s+on\s+campus\b"),
            re.compile(r"\bhealth\s+center\b"),
        ],
    ),
    (
        "WIFI",
        [
            re.compile(r"\b(?:is\s+there\s+)?wi\s*-?\s*fi\b"),
            re.compile(r"\binternet\s+(?:facility|available|connection)\b"),
            re.compile(r"\bcampus\s+wifi\b"),
        ],
    ),
    (
        "BANKING_ATM",
        [
            re.compile(r"\b(?:is\s+there\s+an\s+)?atm\b"),
            re.compile(r"\bcanara\s+bank\b"),
            re.compile(r"\bbank\s+(?:counter|on\s+campus)\b"),
            re.compile(r"\bcash\s+counter\b"),
        ],
    ),
    (
        "TOYOTA_CENTER",
        [
            re.compile(r"\btoyota\s+(?:center|centre)\s+of\s+excellence\b"),
            re.compile(r"\btoyota\s+(?:center|centre|lab)\b"),
        ],
    ),
    (
        "TRANSPORT",
        [
            re.compile(r"\b(?:college\s+)?bus(?:es)?\s*(?:facility|service)?\b"),
            re.compile(r"\b(?:provide\s+)?transport(?:ation)?\b"),
            re.compile(r"\bcollege\s+transport\b"),
        ],
    ),
    (
        "COUNSELLING",
        [
            re.compile(r"\bcounselling\b"),
            re.compile(r"\bcounselor\b"),
            re.compile(r"\bmental\s+health\b"),
            re.compile(r"\bcalendly\b"),
        ],
    ),
    # 5. Admissions, Codes & Fees
    (
        "COMEDK_CODE",
        [
            re.compile(r"\bcomed\s*-?\s*k\s+code\b"),
            re.compile(r"\bcomedk\s+institute\s+code\b"),
        ],
    ),
    (
        "CET_CODE",
        [
            re.compile(r"\b(?:k\s*)?cet\s+code\b"),
            re.compile(r"\bkea\s+code\b"),
            re.compile(r"\bcet\s+institute\s+code\b"),
        ],
    ),
    (
        "ADMISSION_CODES",
        [
            re.compile(r"\badmission\s+codes?\b"),
            re.compile(r"\bcounselling\s+codes?\b"),
            re.compile(r"\bcodes?\s+for\s+admission\b"),
        ],
    ),
    (
        "MANAGEMENT_FEES",
        [
            re.compile(r"\bmanagement\s+(?:quota\s+)?fees?\b"),
            re.compile(r"\bmanagement\s+phase\b"),
            re.compile(r"\bmanagement\s+seat\s+cost\b"),
            re.compile(r"\bmanagement\s+quota\s+fee\s+structure\b"),
        ],
    ),
    (
        "SCHOLARSHIPS",
        [
            re.compile(r"\bscholarships?\b"),
            re.compile(r"\bfee\s+concessions?\b"),
            re.compile(r"\bfinancial\s+aid\b"),
            re.compile(r"\bnsp\s+scholarship\b"),
        ],
    ),
    (
        "FEES",
        [
            re.compile(r"\b(?:college\s+|b\.?e\.?\s+)?fees?\s+(?:structure|amount|details)?\b"),
            re.compile(r"\bhow\s+much\s+(?:is\s+the\s+)?(?:fee|fees)\b"),
            re.compile(r"\btuition\s+fees?\b"),
            re.compile(r"\bfee\s+structure\b"),
        ],
    ),
    (
        "ADMISSIONS_CONTACT",
        [
            re.compile(r"\b(?:how\s+(?:can\s+i|to)\s+)?contact\s+(?:the\s+)?admissions?\s*(?:department|office)?\b"),
            re.compile(r"\badmissions?\s+(?:phone|email|contact|number)\b"),
            re.compile(r"\badmission\s+helpline\b"),
        ],
    ),
    (
        "ADMISSIONS_ELIGIBILITY",
        [
            re.compile(r"\beligibility\s+(?:criteria\s+)?for\s+(?:admission|b\.?e\.?)\b"),
            re.compile(r"\badmission\s+eligibility\b"),
            re.compile(r"\beligibility\s+to\s+join\b"),
            re.compile(r"\bqualification\s+for\s+admission\b"),
        ],
    ),
    (
        "ADMISSIONS_DOCUMENTS",
        [
            re.compile(r"\bdocuments?\s+required\s+(?:for\s+admission)?\b"),
            re.compile(r"\badmission\s+documents?\b"),
            re.compile(r"\bwhat\s+documents?\s+(?:are\s+)?needed\b"),
            re.compile(r"\bdocument\s+checklist\b"),
        ],
    ),
    (
        "ADMISSIONS_PROCESS",
        [
            re.compile(r"\b(?:how\s+(?:can\s+i|to)\s+get\s+)?admission\s+process\b"),
            re.compile(r"\badmission\s+procedure\b"),
            re.compile(r"\bhow\s+to\s+apply\s+(?:for\s+admission)?\b"),
            re.compile(r"\bprocess\s+steps?\s+for\s+admission\b"),
        ],
    ),
    # 6. Placements
    (
        "PLACEMENTS_2025",
        [
            re.compile(r"\b(?:highest\s+)?package\s+in\s+2025\b"),
            re.compile(r"\bplacements?\s+in\s+2025\b"),
            re.compile(r"\b2025\s+placements?\b"),
            re.compile(r"\b2025\s+highest\s+package\b"),
            re.compile(r"\bhow\s+many\s+placements?\s+in\s+2025\b"),
        ],
    ),
    (
        "PLACEMENTS_HIGHEST",
        [
            re.compile(r"\bhighest\s+(?:placement\s+)?package\b"),
            re.compile(r"\bhighest\s+ctc\b"),
            re.compile(r"\bmaximum\s+package\b"),
            re.compile(r"\btop\s+package\b"),
        ],
    ),
    (
        "PLACEMENTS_OVERVIEW",
        [
            re.compile(r"\b(?:how\s+many\s+)?companies\s+recruit\b"),
            re.compile(r"\brecent\s+recruiters\b"),
            re.compile(r"\bplacement\s+stats?\b"),
            re.compile(r"\bhow\s+are\s+placements?\b"),
            re.compile(r"\btell\s+me\s+about\s+placements?\b"),
            re.compile(r"\bwho\s+recruits\b"),
            re.compile(r"\bplacement\s+record\b"),
            re.compile(r"\bplacements?\b"),
        ],
    ),
    # 7. Departments
    (
        "CSE_INTAKE",
        [
            re.compile(r"\bintake\s+(?:for\s+|of\s+)?(?:the\s+)?cse\b"),
            re.compile(r"\bcse\s+(?:department\s+)?intake\b"),
            re.compile(r"\b(?:how\s+many\s+seats|seat\s+capacity)\s+(?:in|for|of)\s+cse\b"),
            re.compile(r"\bcse\s+seat\s+capacity\b"),
        ],
    ),
    (
        "CSE_DEPT",
        [
            re.compile(r"\bcomputer\s+science\s+(?:and\s+engineering\s+)?department\b"),
            re.compile(r"\bwhere\s+is\s+cse\b"),
            re.compile(r"\bcse\s+block\b"),
            re.compile(r"\bcse\s+department\b"),
        ],
    ),
    (
        "AIML_DEPT",
        [
            re.compile(r"\b(?:cse\s+)?(?:ai\s*ml|aiml|ai\s+and\s+ml)\s+department\b"),
            re.compile(r"\bwhere\s+is\s+(?:ai\s*ml|aiml)\b"),
        ],
    ),
    (
        "DEPARTMENTS_OVERVIEW",
        [
            re.compile(r"\bwhat\s+departments?\s+(?:does\s+rnsit\s+have|are\s+there)\b"),
            re.compile(r"\bdepartments?\s+list\b"),
            re.compile(r"\bbranches\s+offered\b"),
            re.compile(r"\bcourses\s+offered\b"),
            re.compile(r"\bhow\s+many\s+departments?\b"),
        ],
    ),
    (
        "PHD_PROGRAMS",
        [
            re.compile(r"\bph\.?d\.?\s*(?:programs?|courses?|centers?)?\b"),
            re.compile(r"\bdoctoral\s+programs?\b"),
            re.compile(r"\bresearch\s+centers?\b"),
        ],
    ),
    # 8. College General & Meta
    (
        "COLLEGE_LOCATION",
        [
            re.compile(r"\bwhere\s+is\s+rnsit\s*(?:located)?\b"),
            re.compile(r"\brnsit\s+location\b"),
            re.compile(r"\bcollege\s+address\b"),
            re.compile(r"\brnsit\s+address\b"),
            re.compile(r"\bwhere\s+is\s+the\s+college\s+located\b"),
        ],
    ),
    (
        "WEBSITE",
        [
            re.compile(r"\b(?:official\s+)?website\s*(?:of\s+rnsit)?\b"),
            re.compile(r"\brnsit\s+website\b"),
            re.compile(r"\bweb\s+address\b"),
            re.compile(r"\burl\s+of\s+rnsit\b"),
        ],
    ),
    (
        "AFFILIATION",
        [
            re.compile(r"\baffiliation\s*(?:of\s+rnsit)?\b"),
            re.compile(r"\baffiliated\s+to\b"),
            re.compile(r"\bwhich\s+university\b"),
            re.compile(r"\bis\s+rnsit\s+autonomous\b"),
        ],
    ),
    (
        "CLUBS",
        [
            re.compile(r"\bstudent\s+clubs?\b"),
            re.compile(r"\bclubs?\s+(?:at|in)\s+rnsit\b"),
            re.compile(r"\bwhat\s+clubs?\s+(?:are|does|do)\b"),
            re.compile(r"\bwhich\s+clubs?\b"),
            re.compile(r"\bcoding\s+club\b"),
            re.compile(r"\bquizcorp\b"),
            re.compile(r"\badroit\b"),
            re.compile(r"\bclubs?\b"),
        ],
    ),
    (
        "CULTURAL_FEST",
        [
            re.compile(r"\bcultural\s+fest\b"),
            re.compile(r"\baperture\b"),
            re.compile(r"\bcollege\s+fest\b"),
            re.compile(r"\bannual\s+fest\b"),
        ],
    ),
    (
        "CAPABILITIES",
        [
            re.compile(r"\bwhat\s+can\s+you\s+do\b"),
            re.compile(r"\bhow\s+can\s+you\s+help\b"),
            re.compile(r"\bwho\s+are\s+you\b"),
        ],
    ),
]


def detect_entity(query: str) -> DetectedEntity | None:
    """
    Deterministically maps a user query to its canonical entity using
    phrase normalization, aliases, and exact regex rules against college_info.json.
    
    Returns a DetectedEntity with confidence=1.0 on a direct/pattern match,
    or None if no confident rule matched.
    """
    if not query or not query.strip():
        return None

    clean_query = normalize_query(query).lower()

    # Rule matching
    for entity_id, patterns in _ENTITY_RULES:
        for pat in patterns:
            if pat.search(clean_query):
                entity_info = VERIFIED_ENTITIES.get(entity_id)
                if entity_info:
                    logger.info(
                        "[ENTITY MATCH] query=%r -> entity=%s pattern=%r",
                        query, entity_id, pat.pattern
                    )
                    return DetectedEntity(
                        entity_id=entity_id,
                        canonical_name=entity_info["name"],
                        verified_answer=entity_info["answer"],
                        confidence=1.0,
                        entity_type=entity_info.get("type", "general"),
                    )

    return None
