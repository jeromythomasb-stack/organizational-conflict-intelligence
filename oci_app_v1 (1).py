# ============================================================
# ORGANIZATIONAL CONFLICT INTELLIGENCE SYSTEM — V0.6
# Human-to-Human Communication Prototype
#
# Purpose:
#   Detect early communication patterns associated with
#   workplace conflict, explain the evidence, connect patterns
#   to organizational-psychology theories, estimate escalation
#   evidence, and suggest preventive (non-disciplinary)
#   interventions.
#
# IMPORTANT:
#   This is a research prototype, NOT a diagnostic or employee-
#   surveillance system. It does not determine who is right,
#   who is guilty, whether a person is "toxic", or whether a
#   disciplinary action should be taken.
#
#   AI detects and explains. Humans decide.
#
# V0.4 design principles:
#   1. Analyze communication BETWEEN PEOPLE, not isolated keywords.
#   2. Preserve the distinction between disagreement and conflict.
#   3. Team/group references alone are NOT interdepartmental conflict.
#   4. Use sentence-level semantic evidence + explicit linguistic
#      evidence + context.
#   5. Detect signals before assigning a conflict class.
#   6. Separate conflict evidence from escalation evidence.
#   7. Psychological indicators are "possible" organizational
#      indicators, never diagnoses or mind-reading.
#   8. Risk is an evidence index, NOT a probability.
#   9. Explain exactly why a signal was detected.
#  10. Prefer abstention when evidence is weak or ambiguous.
# ============================================================

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# ------------------------------------------------------------
# PAGE
# ------------------------------------------------------------

st.set_page_config(
    page_title="Organizational Conflict Intelligence",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Organizational Conflict Intelligence")
st.caption(
    "Research prototype for early detection and prevention of "
    "workplace conflict from human-to-human communication."
)

st.warning(
    "Research prototype only. This system provides decision support; "
    "it does not diagnose employees, determine guilt, rank people, "
    "make disciplinary decisions, or replace HR/manager judgment."
)


# ------------------------------------------------------------
# THEORY KNOWLEDGE BASE
# ------------------------------------------------------------

THEORIES = {
    "Organizational Justice Theory": {
        "description": (
            "Explains how perceptions of fairness in procedures, "
            "decisions, resource allocation, and treatment can affect "
            "organizational relationships and reactions."
        ),
        "signals": {
            "Perceived unfairness",
            "Communication breakdown",
            "Trust deterioration",
        },
    },
    "Social Identity Theory": {
        "description": (
            "Explains how people can categorize themselves and others "
            "into groups, potentially producing in-group/out-group "
            "distinctions and intergroup tension."
        ),
        "signals": {
            "Interdepartmental tension",
            "Perceived unfairness",
        },
    },
    "Psychological Safety": {
        "description": (
            "Concerns whether people feel able to speak up, disagree, "
            "ask questions, or report problems without interpersonal "
            "or social consequences."
        ),
        "signals": {
            "Psychological safety concern",
            "Communication breakdown",
        },
    },
    "Social Exchange Theory / Trust": {
        "description": (
            "Emphasizes reciprocity, reliability, expectations, and "
            "trust within workplace relationships."
        ),
        "signals": {
            "Trust deterioration",
            "Communication breakdown",
            "Interdepartmental tension",
        },
    },
    "Conflict Management Theory / Thomas-Kilmann": {
        "description": (
            "Provides a framework for understanding responses to "
            "interpersonal conflict, including competing, avoiding, "
            "accommodating, compromising, and collaborating."
        ),
        "signals": {
            "Relationship conflict",
            "Process conflict",
            "Escalating disagreement",
        },
    },
    "Affective Events Theory": {
        "description": (
            "Links workplace events and interactions with affective "
            "responses that can influence attitudes and behavior."
        ),
        "signals": {
            "Relationship conflict",
            "Perceived unfairness",
            "Escalating disagreement",
        },
    },
    "Leader-Member Exchange Theory": {
        "description": (
            "Focuses on differences in leader-member relationships "
            "and perceptions of unequal treatment or support."
        ),
        "signals": {
            "Perceived unfairness",
            "Trust deterioration",
        },
    },
    "Emotional Intelligence": {
        "description": (
            "Provides a lens for understanding interpersonal emotion, "
            "self-regulation, empathy, and relationship management."
        ),
        "signals": {
            "Relationship conflict",
            "Escalating disagreement",
            "Communication breakdown",
        },
    },
}


# ------------------------------------------------------------
# SIGNAL KNOWLEDGE BASE
# ------------------------------------------------------------
#
# Each signal has:
#   description
#   prototypes
#   keywords
#   cue groups
#   theories
#   base_weight
#
# The important design change is that "group language" is NOT a
# conflict signal by itself.
# ------------------------------------------------------------

SIGNALS = {
    "Perceived unfairness": {
        "description": (
            "Communication expressing that decisions, treatment, "
            "opportunities, resources, workload, recognition, or "
            "procedures are unfair, unequal, inconsistent, or unjustified."
        ),
        "prototypes": [
            "I feel that the way work is assigned is unfair.",
            "The criteria for these decisions do not seem fair.",
            "Our contributions are not being recognized equally.",
            "Some people appear to receive better opportunities than others.",
            "The workload is being distributed unequally.",
            "I do not understand why the decision was made differently for us.",
            "The process for assigning responsibilities does not seem consistent.",
            "It feels like we are being treated differently from the others.",
            "The allocation of important projects does not seem equitable.",
        ],
        "keywords": [
            "unfair",
            "unfairly",
            "unjust",
            "inequitable",
            "unequal",
            "unequally",
            "favoritism",
            "favoured",
            "favored",
            "discriminat",
            "treated differently",
            "not valued",
            "not recognized",
            "not recognised",
            "leftovers",
            "same amount of work",
        ],
        "cue_groups": [
            [
                "criteria",
                "process",
                "decision",
                "decision-making",
                "allocation",
                "assigned",
                "assignment",
                "opportunity",
                "opportunities",
                "workload",
                "resources",
                "recognition",
                "treatment",
                "project",
                "projects",
                "role",
                "roles",
            ]
        ],
        "theories": [
            "Organizational Justice Theory",
            "Leader-Member Exchange Theory",
        ],
        "base_weight": 1.0,
    },

    "Communication breakdown": {
        "description": (
            "Communication between people or teams is failing through "
            "missing information, ignored messages, unclear responsibilities, "
            "repeated non-response, or inability to coordinate."
        ),
        "prototypes": [
            "Important information is not being shared with us.",
            "People are not responding to repeated requests for information.",
            "We keep asking for updates but nobody responds.",
            "The teams are not communicating effectively.",
            "Responsibilities are unclear and this is causing coordination problems.",
            "Messages are being ignored and work cannot move forward.",
            "We are not receiving the information we need from the other people involved.",
        ],
        "keywords": [
            "not responding",
            "nobody responds",
            "no response",
            "ignored",
            "ignoring",
            "not communicating",
            "communication has broken down",
            "communication breakdown",
            "not being shared",
            "not shared",
            "missing information",
            "unclear responsibilities",
            "no updates",
            "keep asking",
            "keeps asking",
        ],
        "cue_groups": [
            [
                "information",
                "update",
                "updates",
                "message",
                "messages",
                "response",
                "respond",
                "responding",
                "communicate",
                "communication",
                "responsibilities",
                "handover",
                "coordination",
            ]
        ],
        "theories": [
            "Psychological Safety",
            "Organizational Justice Theory",
            "Social Exchange Theory / Trust",
            "Emotional Intelligence",
        ],
        "base_weight": 1.0,
    },

    "Trust deterioration": {
        "description": (
            "Communication indicating reduced confidence in another "
            "person or group because commitments, expectations, reliability, "
            "or cooperation have deteriorated."
        ),
        "prototypes": [
            "We are starting to lose trust in the other team.",
            "I no longer feel confident that they will follow through.",
            "Repeated broken commitments are making it difficult to trust them.",
            "Our confidence in their reliability is declining.",
            "We are becoming less willing to rely on them.",
            "They have not followed through on several commitments.",
        ],
        "keywords": [
            "lose trust",
            "lost trust",
            "trust is declining",
            "trust has declined",
            "do not trust",
            "don't trust",
            "cannot trust",
            "can't trust",
            "not reliable",
            "reliability",
            "broken commitments",
            "failed to follow through",
            "not followed through",
            "confidence is declining",
            "confidence has declined",
            "less willing to rely",
        ],
        "cue_groups": [
            [
                "trust",
                "reliable",
                "reliability",
                "commitment",
                "commitments",
                "follow through",
                "confidence",
                "rely",
                "reliance",
            ]
        ],
        "theories": [
            "Social Exchange Theory / Trust",
            "Organizational Justice Theory",
            "Leader-Member Exchange Theory",
        ],
        "base_weight": 1.0,
    },

    "Psychological safety concern": {
        "description": (
            "Communication suggesting that people feel unable to speak "
            "up, disagree, ask questions, admit problems, or raise concerns "
            "without fear of negative interpersonal consequences."
        ),
        "prototypes": [
            "People are afraid to speak up about problems.",
            "People do not feel safe disagreeing with the manager.",
            "Employees are worried about how disagreement will be received.",
            "People are staying silent because they fear negative consequences.",
            "Team members do not feel comfortable raising concerns.",
            "It is difficult to admit mistakes because people are afraid of the reaction.",
        ],
        "keywords": [
            "afraid to speak",
            "afraid of speaking",
            "fear speaking",
            "fear of speaking",
            "not safe to speak",
            "not comfortable speaking",
            "cannot speak up",
            "can't speak up",
            "do not feel safe",
            "don't feel safe",
            "worried about speaking",
            "fear negative consequences",
            "staying silent",
            "afraid to disagree",
            "fear disagreement",
        ],
        "cue_groups": [
            [
                "speak",
                "speaking",
                "speak up",
                "disagree",
                "disagreement",
                "concern",
                "concerns",
                "mistake",
                "mistakes",
                "safe",
                "comfortable",
                "silent",
            ]
        ],
        "theories": [
            "Psychological Safety",
            "Emotional Intelligence",
        ],
        "base_weight": 1.05,
    },

    "Interdepartmental tension": {
        "description": (
            "Communication indicating tension, competition, blame, "
            "resentment, distrust, or conflict BETWEEN identifiable "
            "workplace groups, teams, departments, or functions."
        ),
        "prototypes": [
            "There is growing tension between our team and their team.",
            "The two departments are increasingly competing against each other.",
            "Our teams are blaming each other for the problem.",
            "The departments are becoming increasingly hostile toward one another.",
            "There is resentment between the two teams.",
            "The teams are in conflict over important projects.",
            "The relationship between the departments is deteriorating.",
            "Our team and their team keep disagreeing about responsibility.",
        ],
        "keywords": [
            "tension between",
            "conflict between",
            "competing against",
            "competing with",
            "blaming each other",
            "blame each other",
            "resentment between",
            "hostile toward",
            "hostile towards",
            "against each other",
            "teams are in conflict",
            "departments are in conflict",
            "relationship between the teams",
            "deteriorating between",
        ],
        "group_terms": [
            "our team",
            "their team",
            "my team",
            "your team",
            "the other team",
            "both teams",
            "two teams",
            "another team",
            "department",
            "departments",
            "group",
            "groups",
            "function",
            "functions",
            "division",
            "divisions",
            "colleagues",
        ],
        "tension_terms": [
            "tension",
            "conflict",
            "competition",
            "competing",
            "blame",
            "blaming",
            "resentment",
            "hostile",
            "hostility",
            "friction",
            "dispute",
            "deteriorating",
            "deterioration",
            "against each other",
            "clash",
        ],
        "theories": [
            "Social Identity Theory",
            "Social Exchange Theory / Trust",
        ],
        "base_weight": 1.15,
    },

    "Relationship conflict": {
        "description": (
            "Conflict centered on personal relationships, interpersonal "
            "hostility, resentment, personal attacks, or deteriorating "
            "relations between identifiable people."
        ),
        "prototypes": [
            "The disagreement has become personal.",
            "There is growing resentment between the colleagues.",
            "The colleagues are increasingly hostile toward one another.",
            "The relationship between the two people has deteriorated.",
            "They are having personal conflicts rather than discussing the work issue.",
            "The discussion has turned into personal attacks.",
        ],
        "keywords": [
            "become personal",
            "personal attack",
            "personal attacks",
            "personally attacking",
            "resentment",
            "hostile toward",
            "hostile towards",
            "hostility",
            "personal conflict",
            "personal conflicts",
            "relationship has deteriorated",
            "relationship deteriorated",
            "cannot work with",
            "can't work with",
        ],
        "person_terms": [
            "he",
            "she",
            "they",
            "colleague",
            "colleagues",
            "manager",
            "supervisor",
            "employee",
            "employees",
            "coworker",
            "coworkers",
            "person",
            "people",
        ],
        "theories": [
            "Conflict Management Theory / Thomas-Kilmann",
            "Affective Events Theory",
            "Emotional Intelligence",
        ],
        "base_weight": 1.2,
    },

    "Process conflict": {
        "description": (
            "Disagreement or friction about tasks, responsibilities, "
            "procedures, roles, decision processes, priorities, or how "
            "work should be performed."
        ),
        "prototypes": [
            "We disagree about how the work should be done.",
            "The teams disagree about who is responsible for the task.",
            "There is ongoing disagreement about the process.",
            "We cannot agree on the project responsibilities.",
            "The decision process is creating repeated disagreement.",
            "People disagree about priorities and how tasks should be allocated.",
        ],
        "keywords": [
            "disagree about the process",
            "disagreement about the process",
            "disagree about how",
            "disagree on how",
            "responsibility",
            "responsibilities",
            "who is responsible",
            "roles",
            "role",
            "procedure",
            "procedures",
            "process",
            "priorities",
            "task allocation",
            "how the work should be done",
        ],
        "cue_groups": [
            [
                "process",
                "procedure",
                "responsibility",
                "responsibilities",
                "role",
                "roles",
                "task",
                "tasks",
                "priority",
                "priorities",
                "allocation",
                "work",
                "decision",
            ]
        ],
        "theories": [
            "Conflict Management Theory / Thomas-Kilmann",
            "Organizational Justice Theory",
        ],
        "base_weight": 0.95,
    },

    "Escalating disagreement": {
        "description": (
            "A disagreement that is becoming more intense, repeated, "
            "persistent, hostile, difficult to resolve, or damaging to "
            "ongoing cooperation."
        ),
        "prototypes": [
            "This is the third time we have raised the same issue and nothing has changed.",
            "The disagreement is becoming increasingly hostile.",
            "People are becoming increasingly frustrated about the unresolved issue.",
            "The same problem keeps happening and cooperation is breaking down.",
            "The issue has persisted despite repeated attempts to resolve it.",
            "If this continues, the teams may no longer be able to work together.",
        ],
        "escalation_prototypes": [
            "This is the third time we have raised the same issue.",
            "Nothing has changed despite repeated attempts to resolve the problem.",
            "The issue keeps happening again and again.",
            "People are becoming increasingly frustrated.",
            "The disagreement is becoming more hostile.",
            "Cooperation is starting to break down.",
            "If this continues, we may not be able to work together.",
            "The conflict is getting worse.",
            "The situation is escalating.",
        ],
        "keywords": [
            "third time",
            "again and again",
            "keeps happening",
            "keeps occurring",
            "repeatedly",
            "repeated attempts",
            "nothing has changed",
            "still unresolved",
            "increasingly frustrated",
            "increasing frustration",
            "becoming more hostile",
            "getting worse",
            "escalating",
            "escalation",
            "cooperation is breaking down",
            "cooperation is starting to break down",
            "cannot keep working together",
            "can't keep working together",
            "no longer work together",
        ],
        "theories": [
            "Conflict Management Theory / Thomas-Kilmann",
            "Affective Events Theory",
            "Emotional Intelligence",
        ],
        "base_weight": 1.25,
    },
}

SIGNAL_NAMES = list(SIGNALS.keys())


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

# Hugging Face model repository containing the frozen OCI DistilBERT baseline.
# Replace YOUR_USERNAME with your actual Hugging Face username.
HF_MODEL_ID = "germibird/oci-distilbert-v1"

LABELS = {
    0: "No conflict",
    1: "Constructive disagreement",
    2: "Emerging concern",
    3: "Active conflict",
    4: "Escalating conflict",
}

@st.cache_resource(show_spinner="Loading semantic language model…")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading OCI DistilBERT classifier…")
def load_classifier():
    if HF_MODEL_ID.startswith("YOUR_USERNAME/"):
        st.error("Please replace YOUR_USERNAME in HF_MODEL_ID with your Hugging Face username.")
        st.stop()
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    classifier = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    classifier.eval()
    return tokenizer, classifier

model = load_model()
tokenizer, classifier = load_classifier()

def predict_conflict_state(text: str) -> Tuple[str, float, Dict[str, float]]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = classifier(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    pred_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[pred_id].item())
    probability_map = {
        LABELS[i]: float(probabilities[i].item())
        for i in range(len(LABELS))
    }
    return LABELS[pred_id], confidence, probability_map


# ------------------------------------------------------------
# EMBEDDING INDEX
# ------------------------------------------------------------

@st.cache_resource(show_spinner="Building signal knowledge base…")
def build_prototype_index():
    texts = []
    owners = []

    for name in SIGNAL_NAMES:
        data = SIGNALS[name]

        for prototype in data.get("prototypes", []):
            texts.append(prototype)
            owners.append(name)

        for prototype in data.get("escalation_prototypes", []):
            texts.append(prototype)
            owners.append(name)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings), owners, texts


PROTOTYPE_EMBEDDINGS, PROTOTYPE_OWNERS, PROTOTYPE_TEXTS = build_prototype_index()


# ------------------------------------------------------------
# TEXT UTILITIES
# ------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)

    if not text:
        return []

    # Conservative sentence splitter.
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if s.strip()]


def lower(text: str) -> str:
    return normalize_text(text).lower()


def phrase_in_text(text: str, phrase: str) -> bool:
    return phrase.lower() in lower(text)


def contains_any(text: str, terms: List[str]) -> List[str]:
    t = lower(text)
    return [term for term in terms if term.lower() in t]


def word_boundary_present(text: str, term: str) -> bool:
    pattern = r"\b" + re.escape(term.lower()) + r"\b"
    return bool(re.search(pattern, lower(text)))


def negation_near(text: str, phrase: str) -> bool:
    """
    Conservative negation handling.
    It does NOT attempt full linguistic parsing.
    It only blocks obvious constructions such as:
      "no tension"
      "not a conflict"
      "there is no resentment"
      "we are not hostile"
    """
    t = lower(text)
    p = phrase.lower()

    idx = t.find(p)
    while idx >= 0:
        prefix = t[max(0, idx - 35):idx]

        neg_patterns = [
            r"\bno\s*$",
            r"\bnot\s+(?:a|an|any)?\s*$",
            r"\bwithout\s*$",
            r"\bnever\s*$",
            r"\bnot\s+experiencing\s*$",
            r"\bnot\s+having\s*$",
        ]

        if any(re.search(pattern, prefix) for pattern in neg_patterns):
            return True

        idx = t.find(p, idx + 1)

    return False


# ------------------------------------------------------------
# HUMAN-TO-HUMAN CONTEXT
# ------------------------------------------------------------

def extract_human_actors(text: str) -> Dict[str, List[str]]:
    """
    Extracts broad relationship/actor cues.
    This is deliberately not person identification.
    The system does not name, rank, or profile individuals.
    """
    t = lower(text)

    people = [
        "i",
        "we",
        "me",
        "us",
        "they",
        "them",
        "he",
        "she",
        "him",
        "her",
        "manager",
        "supervisor",
        "employee",
        "employees",
        "colleague",
        "colleagues",
        "coworker",
        "coworkers",
        "people",
        "team",
        "teams",
        "department",
        "departments",
    ]

    relationship = [
        "between",
        "toward",
        "towards",
        "with",
        "from",
        "against",
        "our team",
        "their team",
        "other team",
        "colleague",
        "colleagues",
        "coworker",
        "coworkers",
        "department",
        "departments",
        "manager",
        "supervisor",
    ]

    return {
        "people": [p for p in people if word_boundary_present(t, p)],
        "relationship": [p for p in relationship if p in t],
    }


def has_group_reference(text: str) -> bool:
    t = lower(text)
    group_terms = SIGNALS["Interdepartmental tension"]["group_terms"]
    return any(term.lower() in t for term in group_terms)


def has_intergroup_tension_context(text: str) -> Tuple[bool, List[str]]:
    """
    Critical guard:
      "our team handles testing and their team handles reports"
    => group reference, but NO conflict.

    Strong evidence requires a group reference AND a tension/competition/
    blame/relationship deterioration cue, OR very strong semantic evidence.
    """
    t = lower(text)

    group_evidence = contains_any(
        t,
        SIGNALS["Interdepartmental tension"]["group_terms"],
    )

    tension_evidence = contains_any(
        t,
        SIGNALS["Interdepartmental tension"]["tension_terms"],
    )

    direct_phrases = contains_any(
        t,
        SIGNALS["Interdepartmental tension"]["keywords"],
    )

    if not group_evidence:
        return False, []

    # Explicit negation blocks a conflict interpretation.
    for term in tension_evidence + direct_phrases:
        if negation_near(t, term):
            return False, []

    if direct_phrases:
        return True, list(dict.fromkeys(group_evidence + direct_phrases))

    if tension_evidence:
        return True, list(dict.fromkeys(group_evidence + tension_evidence))

    # Group language without tension is intentionally NOT enough.
    return False, group_evidence


# ------------------------------------------------------------
# EXPLICIT RULE EVIDENCE
# ------------------------------------------------------------

def explicit_signal_evidence(
    signal_name: str,
    text: str,
) -> Dict:
    data = SIGNALS[signal_name]
    t = lower(text)

    evidence = []
    cue_evidence = []

    # Exact signal keywords.
    for keyword in data.get("keywords", []):
        if keyword.lower() in t and not negation_near(t, keyword):
            evidence.append(keyword)

    # Contextual cue groups.
    for group in data.get("cue_groups", []):
        present = contains_any(t, group)
        if present:
            cue_evidence.extend(present)

    # V0.6 targeted procedural-justice evidence.
    # Detects indirect fairness language (criteria, transparency, consistency)
    # without requiring the literal word "unfair". This is evidence of a
    # fairness concern, not automatic evidence of process conflict.
    if signal_name == "Perceived unfairness":
        fairness_patterns = [
            r"\bcriteria\b.*\b(explain|explained|clarif|clear|consisten|applied)\w*\b",
            r"\b(explain|explained|clarif\w*|transparent|transparency)\b.*\b(decision|decisions|criteria|process|promotion|selection|assignment)\b",
            r"\b(lack of|lack|without|not enough|no)\b.*\b(transparency|explanation|clarity)\b",
            r"\b(decision|decisions|criteria|process)\b.*\b(unclear|inconsistent|inconsistently|transparent|transparency)\b",
            r"\b(applied|apply|application)\b.*\b(consistently|consistency|criteria)\b",
        ]
        fairness_hits = [
            pat for pat in fairness_patterns
            if re.search(pat, t, flags=re.IGNORECASE)
        ]
        if fairness_hits:
            evidence.extend(["procedural fairness", "decision transparency"])

    # Special interdepartmental guard.
    if signal_name == "Interdepartmental tension":
        valid, context = has_intergroup_tension_context(text)

        if valid:
            evidence.extend(context)
        else:
            # Remove bare team/group references from explicit evidence.
            evidence = [
                x for x in evidence
                if x.lower() not in SIGNALS[signal_name]["group_terms"]
            ]

    # Relationship conflict needs interpersonal content.
    if signal_name == "Relationship conflict":
        person_terms = contains_any(
            t,
            SIGNALS[signal_name].get("person_terms", []),
        )

        # Personal phrases can establish interpersonal context directly.
        personal_phrase = any(
            phrase.lower() in t
            for phrase in [
                "become personal",
                "personal attack",
                "personal attacks",
                "personal conflict",
                "personal conflicts",
            ]
        )

        if not person_terms and not personal_phrase:
            evidence = []

    # Process conflict should not fire merely because "work" appears.
    if signal_name == "Process conflict":
        if not evidence and len(cue_evidence) < 2:
            evidence = []

    # Escalation needs actual escalation content.
    if signal_name == "Escalating disagreement":
        if not evidence:
            evidence = []

    return {
        "explicit": list(dict.fromkeys(evidence)),
        "cues": list(dict.fromkeys(cue_evidence)),
    }


# ------------------------------------------------------------
# SEMANTIC ANALYSIS
# ------------------------------------------------------------

def semantic_signal_evidence(text: str) -> Dict[str, Dict]:
    sentences = split_sentences(text)

    if not sentences:
        return {
            name: {
                "max_similarity": 0.0,
                "best_sentence": "",
                "prototype": "",
            }
            for name in SIGNAL_NAMES
        }

    sentence_embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sentence_embeddings = np.asarray(sentence_embeddings)

    result = {}

    for name in SIGNAL_NAMES:
        indices = [
            i for i, owner in enumerate(PROTOTYPE_OWNERS)
            if owner == name
        ]

        if not indices:
            result[name] = {
                "max_similarity": 0.0,
                "best_sentence": "",
                "prototype": "",
            }
            continue

        prototype_matrix = PROTOTYPE_EMBEDDINGS[indices]

        similarities = sentence_embeddings @ prototype_matrix.T

        best_flat = int(np.argmax(similarities))
        sentence_idx, proto_idx = np.unravel_index(
            best_flat,
            similarities.shape,
        )

        score = float(similarities[sentence_idx, proto_idx])

        result[name] = {
            "max_similarity": score,
            "best_sentence": sentences[sentence_idx],
            "prototype": PROTOTYPE_TEXTS[indices[proto_idx]],
        }

    return result


# ------------------------------------------------------------
# SIGNAL SCORING
# ------------------------------------------------------------

def score_signal(
    signal_name: str,
    explicit: List[str],
    cues: List[str],
    semantic_score: float,
    text: str,
) -> Dict:
    """
    Evidence fusion.

    Semantic similarity is NOT treated as a probability.

    Conservative thresholds:
      - strong explicit evidence can trigger a signal
      - semantic-only evidence needs a high similarity score
      - ambiguous semantic evidence remains "possible"
    """
    explicit_count = len(explicit)
    cue_count = len(cues)

    # Explicit evidence score.
    if explicit_count >= 2:
        explicit_score = 1.0
    elif explicit_count == 1:
        explicit_score = 0.78
    else:
        explicit_score = 0.0

    # Contextual support.
    cue_score = min(0.25, 0.08 * cue_count)

    # Semantic score converted only into an evidence contribution.
    semantic_contribution = max(
        0.0,
        min(1.0, (semantic_score - 0.35) / 0.45),
    )

    # Base fused evidence.
    evidence = (
        0.50 * explicit_score
        + 0.20 * cue_score
        + 0.30 * semantic_contribution
    )

    # Special rules.
    if signal_name == "Interdepartmental tension":
        valid_group_context, _ = has_intergroup_tension_context(text)

        # Never infer interdepartmental conflict from group labels alone.
        if not valid_group_context:
            if semantic_score >= 0.78 and has_group_reference(text):
                evidence = max(evidence, 0.62)
            else:
                evidence = 0.0

    if signal_name == "Relationship conflict":
        actors = extract_human_actors(text)

        if not actors["people"]:
            evidence *= 0.35

    if signal_name == "Process conflict":
        # A normal disagreement about a method is not automatically conflict.
        # Require either repeated/process-friction wording or multiple cues.
        if explicit_count == 0 and cue_count < 2 and semantic_score < 0.74:
            evidence = min(evidence, 0.25)

    # Healthy disagreement safeguard.
    healthy_markers = [
        "i understand",
        "happy to discuss",
        "open to discuss",
        "willing to discuss",
        "let's discuss",
        "can discuss",
        "another method",
        "alternative",
        "alternatives",
        "respectfully",
        "i see their point",
        "i understand the reasoning",
    ]

    healthy_count = sum(
        1 for marker in healthy_markers if marker in lower(text)
    )

    # Healthy disagreement should suppress weak conflict evidence,
    # but not explicit hostility/escalation.
    if healthy_count >= 2 and signal_name not in {
        "Escalating disagreement",
        "Relationship conflict",
    }:
        evidence *= 0.35

    # Strong explicit evidence receives a floor.
    if explicit_count >= 1:
        evidence = max(evidence, 0.55)

    # Strong semantic evidence without explicit evidence can be "possible".
    if explicit_count == 0 and semantic_score >= 0.76:
        evidence = max(evidence, 0.50)

    # Final strength.
    if evidence >= 0.78:
        strength = "High"
    elif evidence >= 0.56:
        strength = "Medium"
    elif evidence >= 0.42:
        strength = "Low"
    else:
        strength = "Not detected"

    # Confidence reflects agreement between evidence sources.
    source_count = int(explicit_count > 0) + int(semantic_score >= 0.55)

    if source_count >= 2 and evidence >= 0.56:
        confidence = "Higher"
    elif source_count >= 1 and evidence >= 0.50:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return {
        "signal": signal_name,
        "strength": strength,
        "evidence_score": round(float(evidence), 3),
        "semantic_similarity": round(float(semantic_score), 3),
        "explicit_evidence": explicit,
        "contextual_cues": cues,
        "confidence": confidence,
    }


def detect_signals(text: str) -> Dict[str, Dict]:
    semantic = semantic_signal_evidence(text)
    detections = {}

    for name in SIGNAL_NAMES:
        explicit = explicit_signal_evidence(name, text)

        detections[name] = score_signal(
            signal_name=name,
            explicit=explicit["explicit"],
            cues=explicit["cues"],
            semantic_score=semantic[name]["max_similarity"],
            text=text,
        )

        detections[name]["best_sentence"] = semantic[name]["best_sentence"]
        detections[name]["matched_prototype"] = semantic[name]["prototype"]

    return detections


# ------------------------------------------------------------
# HEALTHY / AMBIGUOUS COMMUNICATION SAFEGUARDS
# ------------------------------------------------------------

def detect_healthy_disagreement(text: str) -> bool:
    t = lower(text)

    disagreement = any(
        phrase in t
        for phrase in [
            "i disagree",
            "we disagree",
            "disagree with",
            "different view",
            "another approach",
            "another method",
            "alternative approach",
            "alternative method",
        ]
    )

    constructive = sum(
        phrase in t
        for phrase in [
            "understand",
            "happy to discuss",
            "willing to discuss",
            "open to discuss",
            "let's discuss",
            "discuss the alternatives",
            "respectfully",
            "i see their reasoning",
        ]
    )

    return disagreement and constructive >= 1


def detect_negative_but_nonconflict(text: str) -> bool:
    t = lower(text)

    negative = any(
        x in t
        for x in [
            "difficult",
            "exhausted",
            "tired",
            "stressful",
            "busy",
            "overwhelmed",
            "disappointed",
        ]
    )

    supportive = any(
        x in t
        for x in [
            "supportive",
            "everyone has been supportive",
            "team has been supportive",
            "we are supporting each other",
        ]
    )

    conflict_markers = any(
        x in t
        for x in [
            "hostile",
            "resentment",
            "blaming",
            "blame each other",
            "conflict",
            "tension",
            "do not trust",
            "don't trust",
            "not responding",
            "nothing has changed",
            "unfair",
            "afraid to speak",
        ]
    )

    return negative and supportive and not conflict_markers


# ------------------------------------------------------------
# PSYCHOLOGICAL / ORGANIZATIONAL INDICATORS
# ------------------------------------------------------------

def identify_psychological_indicators(
    detections: Dict[str, Dict],
    text: str,
) -> List[Dict]:
    indicators = []

    mapping = {
        "Perceived unfairness": (
            "Possible perceived organizational injustice",
            "The communication expresses concern about fairness, equality, "
            "decision procedures, treatment, or allocation.",
        ),
        "Communication breakdown": (
            "Possible communication/coordination strain",
            "The communication indicates difficulty exchanging information "
            "or coordinating work with others.",
        ),
        "Trust deterioration": (
            "Possible deterioration of interpersonal or intergroup trust",
            "The communication indicates reduced confidence in another "
            "person or group.",
        ),
        "Psychological safety concern": (
            "Possible psychological safety concern",
            "The communication suggests reluctance or fear around speaking "
            "up, disagreeing, or raising concerns.",
        ),
        "Interdepartmental tension": (
            "Possible intergroup tension",
            "The communication contains evidence of tension or competition "
            "between identifiable workplace groups.",
        ),
        "Relationship conflict": (
            "Possible relationship/interpersonal strain",
            "The communication describes personal hostility, resentment, "
            "or deterioration in a workplace relationship.",
        ),
        "Process conflict": (
            "Possible task/process conflict",
            "The communication concerns disagreement or friction about "
            "roles, procedures, responsibilities, or work processes.",
        ),
        "Escalating disagreement": (
            "Possible escalation pattern",
            "The communication contains evidence that an unresolved "
            "disagreement is repeating, intensifying, or harming cooperation.",
        ),
    }

    for signal, (label, explanation) in mapping.items():
        if detections[signal]["strength"] in {"High", "Medium"}:
            indicators.append(
                {
                    "indicator": label,
                    "explanation": explanation,
                }
            )

    # Healthy disagreement should not create a psychological problem label.
    if detect_healthy_disagreement(text):
        indicators = [
            x for x in indicators
            if "conflict" not in x["indicator"].lower()
            and "escalation" not in x["indicator"].lower()
        ]

    return indicators


# ------------------------------------------------------------
# CONFLICT CLASSIFICATION
# ------------------------------------------------------------

def classify_conflict(
    detections: Dict[str, Dict],
    text: str,
) -> Tuple[str, str]:
    """
    Classification happens AFTER signal detection.

    Important:
      - A signal does not automatically equal a conflict class.
      - A fairness concern alone can be an early warning without
        being classified as a conflict.
      - Healthy disagreement is explicitly separated.
    """

    if detect_negative_but_nonconflict(text):
        return (
            "No clear conflict pattern",
            "Negative experience detected, but the communication also "
            "contains explicit supportive/cooperative context.",
        )

    if detect_healthy_disagreement(text):
        return (
            "Healthy disagreement",
            "A difference of opinion is expressed together with "
            "constructive discussion or willingness to collaborate.",
        )

    def strong(name: str) -> bool:
        return detections[name]["strength"] in {"High", "Medium"}

    # Strong escalation evidence is sufficient to classify the communication
    # as an escalating conflict even when the exact underlying conflict type
    # cannot be established reliably.
    underlying = any(
        strong(name)
        for name in [
            "Perceived unfairness",
            "Communication breakdown",
            "Trust deterioration",
            "Interdepartmental tension",
            "Relationship conflict",
            "Process conflict",
        ]
    )

    if strong("Escalating disagreement"):
        if underlying:
            return (
                "Escalating conflict",
                "An underlying workplace issue is accompanied by evidence "
                "of persistence, intensification, repetition, or declining cooperation.",
            )
        return (
            "Escalating conflict",
            "The communication contains direct evidence of recurrence, "
            "persistence, intensification, or declining cooperation. The "
            "specific underlying conflict type cannot be established reliably "
            "from the available communication.",
        )

    # Relationship conflict has priority over process conflict because
    # personal hostility changes the nature of the interaction.
    if strong("Relationship conflict"):
        return (
            "Relationship conflict",
            "The communication contains evidence of interpersonal "
            "hostility, resentment, personalisation, or relationship strain.",
        )

    # Explicit interdepartmental conflict requires group context.
    if strong("Interdepartmental tension"):
        return (
            "Interdepartmental conflict",
            "The communication contains evidence of tension, competition, "
            "blame, or deterioration between identifiable workplace groups.",
        )

    # A fairness/justice concern should take priority over a generic
    # process classification. A concern about unclear criteria or unequal
    # treatment is an early organizational warning, not necessarily
    # evidence of an actual process conflict.
    if strong("Perceived unfairness"):
        return (
            "Emerging workplace concern",
            "The communication raises a meaningful concern about fairness "
            "or the transparency of organizational decisions, but it does "
            "not provide sufficient evidence for a specific conflict type.",
        )

    # Process conflict requires more than a normal difference of opinion.
    if strong("Process conflict"):
        return (
            "Process conflict",
            "The communication indicates friction or unresolved disagreement "
            "about roles, responsibilities, procedures, or work processes.",
        )

    # Communication problems can be early signals without necessarily
    # being classified as a specific conflict.
    if strong("Communication breakdown"):
        return (
            "Emerging workplace concern",
            "A meaningful organizational signal is present, but the evidence "
            "is not sufficient to classify it as a specific conflict type.",
        )

    if strong("Trust deterioration"):
        return (
            "Emerging relationship concern",
            "Trust-related strain is present, but there is insufficient "
            "evidence to classify a specific conflict type.",
        )

    if strong("Psychological safety concern"):
        return (
            "Emerging workplace concern",
            "The communication raises a possible psychological-safety concern "
            "without enough evidence for a specific conflict classification.",
        )

    return (
        "No clear conflict pattern",
        "The available communication does not contain sufficient evidence "
        "for a conflict classification.",
    )


# ------------------------------------------------------------
# ESCALATION EVIDENCE
# ------------------------------------------------------------

ESCALATION_CATEGORIES = {
    "Recurrence": [
        "third time",
        "again and again",
        "repeatedly",
        "repeated attempts",
        "keeps happening",
        "keeps occurring",
        "same issue",
        "same problem",
    ],
    "Persistence": [
        "nothing has changed",
        "still unresolved",
        "continues",
        "continued",
        "persisted",
        "unresolved",
    ],
    "Intensification": [
        "increasingly",
        "becoming more",
        "getting worse",
        "escalating",
        "more hostile",
        "increasing frustration",
        "increasingly frustrated",
    ],
    "Cooperation impact": [
        "cooperation is breaking down",
        "cooperation is starting to break down",
        "cannot keep working together",
        "can't keep working together",
        "no longer work together",
        "unable to work together",
    ],
}


def assess_escalation(
    detections: Dict[str, Dict],
    text: str,
) -> Dict:
    t = lower(text)

    categories = {}

    for category, markers in ESCALATION_CATEGORIES.items():
        found = [
            marker
            for marker in markers
            if marker in t
        ]
        categories[category] = list(dict.fromkeys(found))

    active_categories = [
        category
        for category, found in categories.items()
        if found
    ]

    direct_signal = detections["Escalating disagreement"]

    # Evidence index, NOT probability.
    # V0.6: score independent escalation dimensions rather than letting the
    # same signal be diluted by a simple average. Four converging dimensions
    # (recurrence, persistence, intensification, cooperation impact) should
    # produce High evidence; one isolated marker should remain conservative.
    dimension_names = [
        "Recurrence",
        "Persistence",
        "Intensification",
        "Cooperation impact",
    ]
    dimension_count = sum(1 for name in dimension_names if categories[name])

    if dimension_count >= 4:
        index = 0.85
    elif dimension_count == 3:
        index = 0.75
    elif dimension_count == 2:
        index = 0.55
    elif dimension_count == 1:
        index = 0.25
    elif direct_signal["strength"] == "High":
        index = 0.60
    elif direct_signal["strength"] == "Medium":
        index = 0.35
    else:
        index = 0.0

    index = min(1.0, index)

    if not active_categories and direct_signal["strength"] == "Not detected":
        level = "Insufficient evidence"
    elif index >= 0.70:
        level = "High"
    elif index >= 0.45:
        level = "Moderate"
    elif index >= 0.20:
        level = "Low"
    else:
        level = "Minimal"

    return {
        "level": level,
        "evidence_index": round(index, 3),
        "categories": categories,
    }


# ------------------------------------------------------------
# THEORY MAPPING
# ------------------------------------------------------------

def map_theories(
    detections: Dict[str, Dict],
) -> List[Dict]:
    theory_scores = {}

    for signal_name, result in detections.items():
        if result["strength"] not in {"High", "Medium"}:
            continue

        weight = {
            "High": 2.0,
            "Medium": 1.0,
        }[result["strength"]]

        for theory in SIGNALS[signal_name]["theories"]:
            theory_scores.setdefault(theory, 0.0)
            theory_scores[theory] += weight

    ranked = sorted(
        theory_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {
            "theory": theory,
            "score": round(score, 2),
            "description": THEORIES[theory]["description"],
        }
        for theory, score in ranked[:4]
    ]


# ------------------------------------------------------------
# RECOMMENDATIONS
# ------------------------------------------------------------

def generate_recommendations(
    detections: Dict[str, Dict],
    conflict_class: str,
    escalation: Dict,
) -> List[str]:
    recommendations = []

    def active(name: str) -> bool:
        return detections[name]["strength"] in {"High", "Medium"}

    if active("Perceived unfairness"):
        recommendations.append(
            "Review how the relevant decision, allocation, or role assignment "
            "was made and clarify the criteria transparently."
        )

    if active("Communication breakdown"):
        recommendations.append(
            "Clarify communication responsibilities, information-sharing "
            "channels, response expectations, and ownership of follow-ups."
        )

    if active("Trust deterioration"):
        recommendations.append(
            "Review unmet commitments and establish explicit, trackable "
            "commitments between the people or groups involved."
        )

    if active("Psychological safety concern"):
        recommendations.append(
            "Create a safe, facilitated opportunity for people to raise "
            "concerns and disagree without retaliation or personal attack."
        )

    if active("Interdepartmental tension"):
        recommendations.append(
            "Consider a facilitated cross-team discussion focused on shared "
            "goals, responsibilities, resource allocation, and unresolved issues."
        )

    if active("Relationship conflict"):
        recommendations.append(
            "Use a neutral facilitated conversation focused on the work issue, "
            "interpersonal impact, and mutually acceptable ways of working."
        )

    if active("Process conflict"):
        recommendations.append(
            "Clarify roles, responsibilities, decision rights, priorities, "
            "and the process for resolving future disagreements."
        )

    if active("Escalating disagreement") or escalation["level"] in {"Moderate", "High"}:
        recommendations.append(
            "Because the issue appears persistent or intensifying, consider "
            "earlier human intervention rather than waiting for formal escalation."
        )

    if not recommendations:
        recommendations.append(
            "No specific preventive intervention is indicated from the "
            "available evidence. Continue normal human communication and review."
        )

    # Always keep intervention non-disciplinary.
    return list(dict.fromkeys(recommendations))


# ------------------------------------------------------------
# EXPLAINABILITY
# ------------------------------------------------------------

def generate_explanation(
    detections: Dict[str, Dict],
    conflict_class: str,
    escalation: Dict,
) -> str:
    active = [
        r for r in detections.values()
        if r["strength"] in {"High", "Medium"}
    ]

    if not active:
        return (
            "The system did not find sufficient combined linguistic and "
            "semantic evidence for a specific conflict pattern."
        )

    strongest = sorted(
        active,
        key=lambda r: r["evidence_score"],
        reverse=True,
    )[:3]

    parts = []

    for r in strongest:
        evidence_parts = []

        if r["explicit_evidence"]:
            evidence_parts.append(
                "explicit wording: " + ", ".join(r["explicit_evidence"][:4])
            )

        if r["semantic_similarity"] >= 0.55:
            evidence_parts.append(
                f"semantic similarity={r['semantic_similarity']:.2f}"
            )

        if r["best_sentence"]:
            evidence_parts.append(
                f"relevant sentence: “{r['best_sentence']}”"
            )

        if evidence_parts:
            parts.append(
                f"{r['signal']} ({r['strength']}) — "
                + "; ".join(evidence_parts)
            )

    explanation = (
        "The classification is based on communication evidence rather "
        "than on sentiment alone. "
        + " | ".join(parts)
    )

    if conflict_class == "No clear conflict pattern":
        explanation += (
            " The system abstains from assigning a conflict class because "
            "the available evidence is insufficient."
        )

    if escalation["level"] in {"Moderate", "High"}:
        explanation += (
            f" Escalation evidence is {escalation['level'].lower()} based "
            "on recurrence, persistence, intensification, and/or impact "
            "on cooperation."
        )

    return explanation


# ------------------------------------------------------------
# CONFIDENCE / UNCERTAINTY
# ------------------------------------------------------------

def overall_confidence(
    detections: Dict[str, Dict],
    conflict_class: str,
) -> Tuple[str, str]:
    active = [
        r for r in detections.values()
        if r["strength"] in {"High", "Medium"}
    ]

    if not active:
        return (
            "Low",
            "The system has insufficient evidence for a specific pattern.",
        )

    high = sum(r["strength"] == "High" for r in active)
    moderate_sources = sum(
        r["confidence"] == "Moderate"
        for r in active
    )

    if high >= 1 and moderate_sources >= 1:
        confidence = "Higher"
    elif high >= 1:
        confidence = "Moderate–High"
    else:
        confidence = "Moderate"

    reason = (
        "Confidence reflects agreement between explicit linguistic evidence "
        "and semantic similarity; it is not a probability that the "
        "classification is correct."
    )

    return confidence, reason


# ------------------------------------------------------------
# UI — INPUT
# ------------------------------------------------------------

st.subheader("1. Human-to-Human Communication")

st.write(
    "Enter a simulated workplace communication between people. "
    "For example, an employee describing an interaction with a colleague, "
    "manager, or another team."
)

example = st.selectbox(
    "Quick test case",
    [
        "Custom",
        "Healthy disagreement",
        "Fairness concern",
        "Team language without conflict",
        "Communication breakdown",
        "Trust deterioration",
        "Psychological safety",
        "Interdepartmental tension",
        "Relationship conflict",
        "Escalating disagreement",
        "Negative but not conflict",
        "Ambiguous disagreement",
    ],
)

EXAMPLES = {
    "Healthy disagreement": (
        "I disagree with the proposed approach because I think another "
        "method would reduce the workload. I understand the team's "
        "reasoning though, and I am happy to discuss the alternatives."
    ),
    "Fairness concern": (
        "I am concerned that the criteria used for assigning project roles "
        "have not been explained clearly. Could management clarify how "
        "these decisions are made?"
    ),
    "Team language without conflict": (
        "Our team will prepare the report while their team handles the "
        "testing. We agreed on these responsibilities during today's meeting."
    ),
    "Communication breakdown": (
        "Important information is not being shared with us. We keep asking "
        "for updates, but nobody is responding."
    ),
    "Trust deterioration": (
        "We are starting to lose trust in the other team because they have "
        "not followed through on several commitments."
    ),
    "Psychological safety": (
        "People are afraid to speak up about problems because they are "
        "worried about how disagreement will be received."
    ),
    "Interdepartmental tension": (
        "There is growing tension between our team and their team. "
        "We are increasingly competing against each other for important projects."
    ),
    "Relationship conflict": (
        "The disagreement has become personal. There is growing resentment "
        "between the colleagues and they are increasingly hostile toward one another."
    ),
    "Escalating disagreement": (
        "This is the third time we have raised the same issue and nothing "
        "has changed. People are becoming increasingly frustrated and "
        "cooperation is starting to break down. If this continues, I do not "
        "know how the teams can keep working together."
    ),
    "Negative but not conflict": (
        "The workload has been difficult this week and I am exhausted, "
        "but everyone on the team has been supportive."
    ),
    "Ambiguous disagreement": (
        "I was disappointed with the project decision, but I understand "
        "why management made it and I am willing to discuss it."
    ),
}

default_text = EXAMPLES.get(example, "")

text = st.text_area(
    "Communication",
    value=default_text,
    height=180,
    placeholder=(
        "Example: I am concerned about how the project decision was made. "
        "I would like to understand the criteria and discuss it with the team."
    ),
)

analyze = st.button(
    "🔎 Analyze Communication",
    type="primary",
    use_container_width=True,
)


# ------------------------------------------------------------
# ANALYSIS
# ------------------------------------------------------------

if analyze:

    text = normalize_text(text)

    if len(text) < 15:
        st.error("Please enter a little more communication text.")
        st.stop()

    with st.spinner("Analyzing communication…"):
        detections = detect_signals(text)

        indicators = identify_psychological_indicators(
            detections,
            text,
        )

        evidence_conflict_class, evidence_class_reason = classify_conflict(
            detections,
            text,
        )

        learned_conflict_class, learned_confidence, class_probabilities = predict_conflict_state(
            text
        )

        escalation = assess_escalation(
            detections,
            text,
        )

        theories = map_theories(detections)

        recommendations = generate_recommendations(
            detections,
            evidence_conflict_class,
            escalation,
        )

        explanation = generate_explanation(
            detections,
            evidence_conflict_class,
            escalation,
        )

        confidence, confidence_reason = overall_confidence(
            detections,
            evidence_conflict_class,
        )

    st.divider()

    # --------------------------------------------------------
    # 2. CONFLICT CLASSIFICATION
    # --------------------------------------------------------

    st.subheader("2. Conflict Classification")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "DistilBERT classification",
            learned_conflict_class,
        )

    with c2:
        st.metric(
            "Model confidence",
            f"{learned_confidence * 100:.1f}%",
        )

    with c3:
        st.metric(
            "Escalation evidence",
            escalation["level"],
        )

    st.info(
        "The learned classifier provides the five-state conflict classification. "
        "The V0.6 evidence layer independently analyzes communication signals and "
        "provides transparent theory-based interpretation and escalation evidence."
    )

    with st.expander("Model probability distribution"):
        prob_rows = [
            {"Class": label, "Probability": f"{prob * 100:.1f}%"}
            for label, prob in sorted(
                class_probabilities.items(), key=lambda x: x[1], reverse=True
            )
        ]
        st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)

    st.caption(
        "V0.6 evidence-layer interpretation: " + evidence_conflict_class + ". "
        + evidence_class_reason
    )

    # --------------------------------------------------------
    # 3. COMMUNICATION SIGNALS
    # --------------------------------------------------------

    st.subheader("3. Communication Signals")

    active = [
        r for r in detections.values()
        if r["strength"] != "Not detected"
    ]

    if not active:
        st.success(
            "No communication signal crossed the conservative detection threshold."
        )
    else:
        rows = []

        for r in sorted(
            active,
            key=lambda x: x["evidence_score"],
            reverse=True,
        ):
            rows.append(
                {
                    "Signal": r["signal"],
                    "Strength": r["strength"],
                    "Evidence score": r["evidence_score"],
                    "Semantic similarity": r["semantic_similarity"],
                    "Confidence": r["confidence"],
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # 4. PSYCHOLOGICAL / ORGANIZATIONAL INDICATORS
    # --------------------------------------------------------

    st.subheader("4. Possible Psychological / Organizational Indicators")

    if not indicators:
        st.write(
            "No strong possible organizational-psychology indicator was "
            "identified from the available communication."
        )
    else:
        for item in indicators:
            with st.expander(item["indicator"]):
                st.write(item["explanation"])

    st.caption(
        "These are possible organizational indicators inferred from "
        "communication patterns. They are not psychological diagnoses "
        "or claims about a person's internal mental state."
    )

    # --------------------------------------------------------
    # 5. THEORY-BASED INTERPRETATION
    # --------------------------------------------------------

    st.subheader("5. Theory-Based Interpretation")

    if not theories:
        st.write(
            "No theory was activated because the evidence did not meet "
            "the threshold for a meaningful signal."
        )
    else:
        for theory in theories:
            with st.expander(theory["theory"]):
                st.write(theory["description"])

                related = [
                    s
                    for s, d in detections.items()
                    if d["strength"] in {"High", "Medium"}
                    and theory["theory"] in SIGNALS[s]["theories"]
                ]

                if related:
                    st.write(
                        "**Communication signals supporting this theory:** "
                        + ", ".join(related)
                    )

    # --------------------------------------------------------
    # 6. EXPLAINABILITY
    # --------------------------------------------------------

    st.subheader("6. How Did the System Reach This Conclusion?")

    st.write(explanation)

    st.caption(
        "Semantic similarity is used as supporting evidence, not as "
        "a probability or proof of conflict."
    )

    # Detailed evidence.
    active_for_details = [
        r for r in detections.values()
        if r["strength"] in {"High", "Medium"}
    ]

    for r in sorted(
        active_for_details,
        key=lambda x: x["evidence_score"],
        reverse=True,
    ):
        with st.expander(
            f"{r['signal']} — {r['strength']}"
        ):
            st.write(
                f"**Combined evidence score:** {r['evidence_score']:.3f}"
            )

            st.write(
                f"**Semantic similarity:** "
                f"{r['semantic_similarity']:.3f}"
            )

            if r["explicit_evidence"]:
                st.write(
                    "**Explicit evidence:** "
                    + ", ".join(r["explicit_evidence"])
                )

            if r["contextual_cues"]:
                st.write(
                    "**Contextual cues:** "
                    + ", ".join(r["contextual_cues"])
                )

            if r["best_sentence"]:
                st.write(
                    "**Most relevant sentence:** "
                    + r["best_sentence"]
                )

            if r["matched_prototype"]:
                st.write(
                    "**Closest knowledge-base example:** "
                    + r["matched_prototype"]
                )

    # --------------------------------------------------------
    # 7. ESCALATION DETAILS
    # --------------------------------------------------------

    st.subheader("7. Escalation Evidence")

    st.write(
        "The escalation score is an evidence index, not a probability."
    )

    e1, e2 = st.columns(2)

    with e1:
        st.metric(
            "Evidence index",
            f"{escalation['evidence_index']:.2f}",
        )

    with e2:
        st.metric(
            "Escalation level",
            escalation["level"],
        )

    escalation_rows = []

    for category, found in escalation["categories"].items():
        escalation_rows.append(
            {
                "Escalation dimension": category,
                "Evidence": ", ".join(found) if found else "None",
            }
        )

    st.dataframe(
        pd.DataFrame(escalation_rows),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # 8. PREVENTIVE RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader("8. Preventive Recommendations")

    for rec in recommendations:
        st.write("• " + rec)

    st.info(
        "Recommendations are preventive and non-disciplinary. "
        "Human review is required before any organizational action."
    )

    # --------------------------------------------------------
    # 9. CONFIDENCE & UNCERTAINTY
    # --------------------------------------------------------

    st.subheader("9. Confidence & Uncertainty")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Overall evidence confidence", confidence)

    with c2:
        st.metric(
            "Evidence-based escalation",
            escalation["level"],
        )

    st.write(confidence_reason)

    if confidence in {"Low", "Moderate"}:
        st.warning(
            "The communication should be reviewed by a human because "
            "the available evidence is not strong enough for a highly "
            "confident interpretation."
        )

    # --------------------------------------------------------
    # 10. HUMAN REVIEW
    # --------------------------------------------------------

    st.subheader("10. Human Review")

    st.warning(
        "AI detects and explains. Humans decide.\n\n"
        "A manager, HR professional, or appropriately trained reviewer "
        "should examine the original context before interpreting the "
        "result or taking action."
    )


# ------------------------------------------------------------
# RESEARCH DASHBOARD / LIMITATIONS
# ------------------------------------------------------------

st.divider()

with st.expander("📊 Research Prototype — What This Version Does"):
    st.markdown(
        """
### Pipeline

**Human-to-human communication**
→ **sentence-level semantic analysis**
→ **explicit linguistic evidence**
→ **contextual guards**
→ **communication signals**
→ **possible organizational/psychological indicators**
→ **conflict classification**
→ **escalation evidence**
→ **theory-based interpretation**
→ **preventive recommendation**
→ **human review**

### What makes this different from sentiment analysis?

A message can be negative without representing workplace conflict.

For example:

> "The workload has been difficult this week and I am exhausted, but everyone on the team has been supportive."

This should not automatically become a conflict case.

Likewise:

> "I disagree with the proposed approach, but I understand the reasoning and am happy to discuss alternatives."

This is a disagreement, but it can be constructive.

The system therefore looks for **relationship, fairness, coordination,
trust, safety, intergroup, process, and escalation evidence**, rather
than simply counting negative words.

### Important interdepartmental safeguard

The following should **not** be interpreted as conflict:

> "Our team will prepare the report while their team handles testing."

The presence of "our team" and "their team" only establishes group identity.

Interdepartmental tension requires additional evidence such as:
- tension
- competition
- blame
- resentment
- hostility
- conflict
- deterioration
- repeated intergroup friction

### Current model

This is a **hybrid research baseline**:
- transparent linguistic rules
- contextual safeguards
- sentence-level semantic embeddings
- theory knowledge base
- explainable evidence fusion

It is not yet a trained organizational-conflict classifier.

### What must happen before research claims are made?

The system needs a labeled evaluation dataset.

Recommended evaluation:
- balanced positive/negative/ambiguous examples
- multiple communication styles and paraphrases
- signal-level precision, recall, F1
- conflict-class confusion matrix
- false-positive and false-negative analysis
- robustness tests
- subgroup/fairness analysis where appropriate
- expert annotation
- inter-rater agreement
- calibration only if probabilistic outputs are later introduced
"""
    )

with st.expander("⚖️ Ethics, Privacy & Scope"):
    st.markdown(
        """
### This prototype must NOT be used to:

- diagnose mental health
- label employees as "toxic"
- rank employees by conflict likelihood
- determine guilt or blame
- automatically report employees
- make disciplinary decisions
- infer protected/sensitive personal characteristics
- replace HR or managerial judgment

### Data principle

The prototype is intended to begin with **synthetic/simulated
workplace communication**.

Real employee communication should only be used after appropriate
organizational approval, privacy safeguards, ethical review where
required, data minimization, anonymization/pseudonymization,
secure storage, access controls, and retention policies.

**AI detects and explains. Humans decide.**
"""
    )

with st.expander("🧪 Built-in Validation Cases"):
    validation_cases = {
        "Fairness concern": EXAMPLES["Fairness concern"],
        "Team language without conflict": EXAMPLES["Team language without conflict"],
        "Healthy disagreement": EXAMPLES["Healthy disagreement"],
        "Communication breakdown": EXAMPLES["Communication breakdown"],
        "Trust deterioration": EXAMPLES["Trust deterioration"],
        "Psychological safety": EXAMPLES["Psychological safety"],
        "Interdepartmental tension": EXAMPLES["Interdepartmental tension"],
        "Relationship conflict": EXAMPLES["Relationship conflict"],
        "Escalating disagreement": EXAMPLES["Escalating disagreement"],
        "Negative but not conflict": EXAMPLES["Negative but not conflict"],
        "Ambiguous disagreement": EXAMPLES["Ambiguous disagreement"],
    }

    for name, case in validation_cases.items():
        st.markdown(f"**{name}**")
        st.write(case)

st.caption(
    "Organizational Conflict Intelligence V0.6 • Research prototype • "
    "Human oversight required"
)
