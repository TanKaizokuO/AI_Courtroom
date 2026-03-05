"""
courtroom.py — Main trial orchestration pipeline.

Wires together the RAG retriever and the three GPT agents to run a full
simulated legal trial from a user-supplied case description.
"""

from __future__ import annotations

import textwrap
from rag import RAGRetriever, retrieve_context
from agents import prosecutor_agent, defense_agent, judge_agent

# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_DIVIDER = "=" * 70
_SUB_DIV = "-" * 70


def _banner(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


def _section(label: str, content: str) -> None:
    print(f"\n{_SUB_DIV}")
    print(f"  {label}")
    print(_SUB_DIV)
    print(content)


# ---------------------------------------------------------------------------
# Trial pipeline
# ---------------------------------------------------------------------------

def run_trial(case: str, retriever: RAGRetriever | None = None, top_k: int = 4) -> dict:
    """
    Run a complete AI courtroom simulation for the given *case* description.

    Parameters
    ----------
    case      : Plain-text description of the case and available facts.
    retriever : A pre-built RAGRetriever. If None, a new one is instantiated.
    top_k     : Number of legal passages to retrieve.

    Returns
    -------
    dict with keys: case, legal_context, prosecution, defense, verdict
    """

    if retriever is None:
        retriever = RAGRetriever()

    # ------------------------------------------------------------------
    # Step 1 – Print case summary
    # ------------------------------------------------------------------
    _banner("⚖️  AI COURTROOM SIMULATION  ⚖️")
    print(f"\nCASE:\n{textwrap.indent(case.strip(), '  ')}")

    # ------------------------------------------------------------------
    # Step 2 – RAG retrieval
    # ------------------------------------------------------------------
    _banner("📚  STEP 1: RAG — Retrieving Legal Context")
    legal_context = retrieve_context(case, retriever, top_k=top_k)
    print(f"\nRetrieved {top_k} relevant legal principles:\n{legal_context}")

    # ------------------------------------------------------------------
    # Step 3 – Prosecutor
    # ------------------------------------------------------------------
    _banner("🔴  STEP 2: PROSECUTOR ARGUMENT")
    print("\nGenerating prosecution argument …")
    prosecution = prosecutor_agent(case, legal_context)
    print(prosecution)

    # ------------------------------------------------------------------
    # Step 4 – Defense
    # ------------------------------------------------------------------
    _banner("🔵  STEP 3: DEFENSE ARGUMENT")
    print("\nGenerating defense argument …")
    defense = defense_agent(case, legal_context)
    print(defense)

    # ------------------------------------------------------------------
    # Step 5 – Judge
    # ------------------------------------------------------------------
    _banner("⚖️   STEP 4: JUDGE VERDICT")
    print("\nDeliberating …")
    verdict = judge_agent(case, legal_context, prosecution, defense)
    print(verdict)

    _banner("✅  TRIAL COMPLETE")

    return {
        "case": case,
        "legal_context": legal_context,
        "prosecution": prosecution,
        "defense": defense,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + _DIVIDER)
    print("  Welcome to the AI Courtroom Simulation")
    print("  Type your case description below.")
    print("  (Press Enter twice when done, or type 'demo' for a built-in example)")
    print(_DIVIDER + "\n")

    raw = input("Case description (or 'demo'): ").strip()

    if raw.lower() == "demo":
        case = (
            "A man was seen running away from a jewelry store moments after the alarm "
            "went off. Security camera footage shows a person matching his description "
            "near the scene. No stolen items were found on him. He claims he was "
            "jogging in the area and panicked when he heard the alarm."
        )
        print(f"\n[Using demo case]\n{case}")
    else:
        lines = [raw]
        print("Continue (press Enter on an empty line to finish):")
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        case = "\n".join(lines)

    retriever = RAGRetriever()
    run_trial(case, retriever=retriever)


if __name__ == "__main__":
    main()
