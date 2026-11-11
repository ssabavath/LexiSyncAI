
"""Evaluate dense-only and hybrid retrieval on held-out questions.

API errors, genuine abstentions and generated answers are recorded separately.
Results are saved after every question to avoid losing progress.
"""

from dotenv import load_dotenv
load_dotenv()

import csv
import time
from pathlib import Path

from src.pdf_ingest import process_pdf
from src.retrieval import HybridRetriever
from src.answer import answer_question


PDF_DIR = Path("data/pdfs")
QUESTION_FILE = Path("data/evaluation_questions.csv")
OUT_FILE = Path("data/evaluation_results.csv")

RETRIEVAL_MODES = ["dense", "hybrid"]
TOP_K = 6

# Wait between API requests to reduce the chance of rate limits.
REQUEST_PAUSE_SECONDS = 2

FIELDNAMES = [
    "question_id",
    "retrieval_mode",
    "question",
    "question_type",
    "expected_answer",
    "system_answer",
    "status",
    "confidence",
    "actual_abstention",
    "abstention_correct_0_or_1",
    "evidence_pages",
    "retrieved_chunks",
    "latency_seconds",
    "manual_answer_correct_0_or_1",
    "manual_citation_precision_0_to_1",
    "manual_evidence_recall_at_6_0_to_1",
    "review_notes",
]


def load_evaluation_questions():
    """Load evaluation questions and ignore blank lines."""

    with QUESTION_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        non_empty_lines = (
            line for line in file if line.strip()
        )

        reader = csv.DictReader(non_empty_lines)

        if not reader.fieldnames:
            raise SystemExit(
                "The evaluation CSV is empty or has no header."
            )

        reader.fieldnames = [
            name.strip().lower()
            for name in reader.fieldnames
        ]

        required_columns = {
            "question",
            "question_type",
            "expected_answer",
        }

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise SystemExit(
                f"Missing CSV columns: {missing}"
            )

        questions = []

        for row in reader:
            cleaned = {
                key.strip().lower(): (
                    value.strip() if value else ""
                )
                for key, value in row.items()
                if key is not None
            }

            if cleaned.get("question"):
                questions.append(cleaned)

    print("Questions loaded:", len(questions))

    return questions


def determine_status(result):
    """Identify success, genuine abstention or API failure."""

    # Prefer the explicit status returned by answer_question().
    explicit_status = result.get("status")

    if explicit_status:
        return str(explicit_status).upper()

    # Compatibility with older answer_question() versions.
    answer = str(result.get("answer", "")).lower()

    if (
        "llm generation is not configured or unavailable" in answer
        or "llm connection note:" in answer
        or "api error" in answer
    ):
        return "API_ERROR"

    if (
        result.get("confidence") == "INSUFFICIENT EVIDENCE"
    ):
        return "ABSTAINED"

    return "SUCCESS"


def is_unanswerable(question):
    return (
        question.get("question_type", "").strip().lower()
        == "unanswerable"
    )


def save_results(rows):
    """Save all collected results, including partial progress."""

    OUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUT_FILE.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()
        writer.writerows(rows)


def calculate_summary(rows):
    """Calculate automated metrics without counting API errors as abstentions."""

    print("\n========== AUTOMATED EVALUATION SUMMARY ==========")

    for mode in RETRIEVAL_MODES:

        mode_rows = [
            row for row in rows
            if row["retrieval_mode"] == mode
        ]

        if not mode_rows:
            continue

        successful_rows = [
            row for row in mode_rows
            if row["status"] != "API_ERROR"
        ]

        api_errors = sum(
            row["status"] == "API_ERROR"
            for row in mode_rows
        )

        abstentions = sum(
            row["status"] == "ABSTAINED"
            for row in mode_rows
        )

        answerable_rows = [
            row for row in successful_rows
            if row["question_type"].lower() != "unanswerable"
        ]

        unanswerable_rows = [
            row for row in successful_rows
            if row["question_type"].lower() == "unanswerable"
        ]

        correct_decisions = sum(
            int(row["abstention_correct_0_or_1"])
            for row in successful_rows
            if row["abstention_correct_0_or_1"] != ""
        )

        latency_values = [
            float(row["latency_seconds"])
            for row in mode_rows
        ]

        average_latency = (
            sum(latency_values) / len(latency_values)
        )

        print(f"\nRetrieval mode: {mode.upper()}")
        print("Total questions:", len(mode_rows))
        print("Successful or abstained:", len(successful_rows))
        print("API errors:", api_errors)
        print("Genuine abstentions:", abstentions)

        print(
            "Average latency:",
            round(average_latency, 3),
            "seconds"
        )

        if successful_rows:
            print(
                "Correct abstention decisions:",
                correct_decisions,
                "/",
                len(successful_rows)
            )

        if answerable_rows:
            wrong_abstentions = sum(
                row["status"] == "ABSTAINED"
                for row in answerable_rows
            )

            print(
                "Answerable questions incorrectly abstained:",
                wrong_abstentions,
                "/",
                len(answerable_rows)
            )

        if unanswerable_rows:
            correct_refusals = sum(
                row["status"] == "ABSTAINED"
                for row in unanswerable_rows
            )

            print(
                "Unanswerable questions correctly refused:",
                correct_refusals,
                "/",
                len(unanswerable_rows)
            )

    print("\nManual answer correctness and citation precision")
    print("must be reviewed separately in the CSV.")


def main():

    # 1. Check that the PDF folder contains documents.
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise SystemExit(
            "No PDFs found. Add documents to data/pdfs/."
        )

    if not 2 <= len(pdf_files) <= 5:
        raise SystemExit(
            f"Found {len(pdf_files)} PDFs. "
            "The assignment requires 2–5 PDFs."
        )

    # 2. Extract PDF chunks.
    chunks = []

    print("\n========== PDF INGESTION ==========")

    for pdf in pdf_files:

        print(f"\nProcessing: {pdf.name}")

        extracted, warnings = process_pdf(
            str(pdf),
            pdf.name
        )

        chunks.extend(extracted)

        for warning in warnings:
            print("WARNING:", warning)

    if not chunks:
        raise SystemExit("No text chunks were extracted.")

    print("\nTotal extracted chunks:", len(chunks))

    # 3. Build retrieval system.
    print("\n========== BUILDING RETRIEVER ==========")

    retriever = HybridRetriever(chunks)

    # 4. Load and validate evaluation questions.
    questions = load_evaluation_questions()

    if len(questions) < 15:
        raise SystemExit(
            "The assignment requires at least 15 questions."
        )

    unanswerable_count = sum(
        is_unanswerable(q)
        for q in questions
    )

    if unanswerable_count < 3:
        raise SystemExit(
            "Add at least 3 unanswerable questions."
        )

    print("Unanswerable questions:", unanswerable_count)

    # 5. Run dense and hybrid retrieval.
    rows = []

    for mode in RETRIEVAL_MODES:

        print(
            f"\n========== {mode.upper()} RETRIEVAL =========="
        )

        for index, question in enumerate(questions, start=1):

            question_text = question["question"]

            expected_abstention = is_unanswerable(question)

            print(
                f"\n{mode.upper()} | "
                f"Question {index}/{len(questions)}"
            )

            start = time.time()

            try:
                result = answer_question(
                    question_text,
                    retriever,
                    top_k=TOP_K,
                    mode=mode
                )

                latency = round(time.time() - start, 3)

                status = determine_status(result)

                evidence = result.get("evidence", [])

                answer = result.get("answer", "")

                confidence = result.get("confidence", "")

                # Do not classify API failures as abstentions.
                actual_abstention = (
                    status == "ABSTAINED"
                )

                if status == "API_ERROR":
                    abstention_correct = ""

                else:
                    abstention_correct = int(
                        expected_abstention
                        == actual_abstention
                    )

                evidence_pages = "; ".join(
                    sorted({
                        (
                            f"{item.get('document', 'Unknown')} "
                            f"p.{item.get('page', '?')} "
                            f"({item.get('section', 'Unknown')})"
                        )
                        for item in evidence
                    })
                )

                retrieved_chunks = "\n\n".join(
                    (
                        f"[{i}] "
                        f"{item.get('document', 'Unknown')} | "
                        f"Page {item.get('page', '?')} | "
                        f"{item.get('section', 'Unknown')}\n"
                        f"{item.get('text', '')}"
                    )
                    for i, item in enumerate(evidence, start=1)
                )

                row = {
                    "question_id": index,
                    "retrieval_mode": mode,
                    "question": question_text,
                    "question_type": question["question_type"],
                    "expected_answer": question["expected_answer"],
                    "system_answer": answer,
                    "status": status,
                    "confidence": confidence,
                    "actual_abstention": int(actual_abstention),
                    "abstention_correct_0_or_1": abstention_correct,
                    "evidence_pages": evidence_pages,
                    "retrieved_chunks": retrieved_chunks,
                    "latency_seconds": latency,
                    "manual_answer_correct_0_or_1": "",
                    "manual_citation_precision_0_to_1": "",
                    "manual_evidence_recall_at_6_0_to_1": "",
                    "review_notes": "",
                }

            except Exception as exc:

                latency = round(time.time() - start, 3)

                print("ERROR:", str(exc))

                row = {
                    "question_id": index,
                    "retrieval_mode": mode,
                    "question": question_text,
                    "question_type": question["question_type"],
                    "expected_answer": question["expected_answer"],
                    "system_answer": f"Evaluation error: {str(exc)}",
                    "status": "API_ERROR",
                    "confidence": "",
                    "actual_abstention": "",
                    "abstention_correct_0_or_1": "",
                    "evidence_pages": "",
                    "retrieved_chunks": "",
                    "latency_seconds": latency,
                    "manual_answer_correct_0_or_1": "",
                    "manual_citation_precision_0_to_1": "",
                    "manual_evidence_recall_at_6_0_to_1": "",
                    "review_notes": "Review error before interpreting result.",
                }

            rows.append(row)

            # Save after every question to protect completed results.
            save_results(rows)

            print(
                f"Status: {row['status']} | "
                f"Latency: {latency}s"
            )

            if row["status"] == "API_ERROR":
                print(
                    "This question encountered an error. "
                    "It is not counted as an abstention."
                )

            time.sleep(REQUEST_PAUSE_SECONDS)

    # 6. Print summary.
    calculate_summary(rows)

    print(
        f"\nSaved {len(rows)} evaluation rows to:"
        f"\n{OUT_FILE.resolve()}"
    )

    print(
        "\nNext: manually review the answers, citations "
        "and evidence relevance in the CSV."
    )


if __name__ == "__main__":
    main()
