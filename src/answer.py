
"""Evidence-grounded legal answer generation with retries and abstention."""

import os
import re
import time
import requests


ABSTAIN_TEXT = (
    "The supplied documents do not contain enough evidence "
    "to answer this question reliably."
)

ABSTAIN_MARKER = "[ABSTAIN]"


def format_evidence(evidence):
    """Format retrieved passages with stable source identifiers."""

    formatted_sources = []

    for index, item in enumerate(evidence, start=1):

        document = item.get("document", "Unknown document")
        page = item.get("page", "Unknown page")
        section = item.get("section", "Unknown section")
        paragraph = item.get("paragraph")

        source_header = (
            f"[Source {index}]\n"
            f"Document: {document}\n"
            f"Physical PDF page: {page}\n"
            f"Article/section: {section}\n"
        )

        if paragraph:
            source_header += f"Paragraph: {paragraph}\n"

        source_header += "Evidence text:\n"

        source_text = str(item.get("text", "")).strip()

        formatted_sources.append(
            source_header + source_text
        )

    return "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(
        formatted_sources
    )


def call_llm(question, evidence, max_attempts=4):
    """
    Send a question and retrieved evidence to the LLM.

    Retry temporary API failures and rate limits.
    """

    api_key = os.getenv("LLM_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is missing from the .env file."
        )

    base_url = os.getenv(
        "LLM_BASE_URL",
        "https://api.groq.com/openai/v1"
    ).rstrip("/")

    model = os.getenv(
        "LLM_MODEL",
        "openai/gpt-oss-120b"
    )

    context = format_evidence(evidence)

    system = """
You are LexiSyncAI, an evidence-grounded legal document
analysis assistant.

Your task is to answer questions using ONLY the evidence
provided in the user message.

STRICT EVIDENCE RULES:

1. Use only the supplied passages.
2. Never use outside legal knowledge, memory, assumptions,
   or information from the internet.
3. Do not invent legal requirements, article numbers,
   paragraph numbers, quotations, statistics, or citations.
4. Every substantive factual or legal claim must have
   an appropriate citation.
5. A citation must identify the exact source number,
   document name, physical PDF page, and article or section
   when that information is available.
6. Cite a source only when its actual passage supports
   the statement being made.
7. Do not cite a passage merely because it mentions
   a related topic.
8. Do not claim that a legal provision imposes an obligation
   unless the supplied passage supports that interpretation.
9. Clearly distinguish what the GDPR says from what the
   EU AI Act says.
10. Do not describe two legal obligations as identical
    simply because they are related.
11. Do not introduce additional article references in the
    conclusion unless those articles appear in the supplied
    evidence and support the relevant claim.
12. Do not invent missing information to make an answer
    appear complete.

CROSS-DOCUMENT QUESTIONS:

When a question requires evidence from multiple documents:

- Identify which document supports each claim.
- Explain the relationship between the documents only
  when the supplied passages establish that relationship.
- Keep the requirements of each regulation separate.
- If evidence from one document is missing, say so.
- Do not pretend that a passage from one regulation
  proves a requirement in another regulation.

ABSTENTION:

If the supplied evidence is insufficient to answer the
question reliably, begin your entire response with:

[ABSTAIN]

Then briefly explain what information is missing.

Do not guess or fill evidence gaps using outside knowledge.

ANSWER FORMAT:

- Answer the question directly.
- Use headings and numbered lists when helpful.
- Use tables only when they make comparisons clearer.
- Cite each substantive legal claim immediately after
  the statement it supports.
- Use this citation format:

  [Source 1, GDPR.pdf, p. 6, Article 5(1)(a)]

- Use the exact source numbers provided in the evidence.
- Do not create new source numbers.
- Do not add a bibliography containing sources that
  were not supplied.

IMPORTANT:

A fluent answer is not necessarily a correct answer.
Evidence accuracy is more important than answer length.

This is an educational document-analysis tool,
not a substitute for professional legal advice.
"""

    user_message = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED EVIDENCE:\n{context}\n\n"
        "Answer the question using only the retrieved evidence. "
        "Check that each citation supports the claim immediately "
        "before it. If the evidence is insufficient, abstain."
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    url = f"{base_url}/chat/completions"

    # Retry temporary failures, including HTTP 429.
    for attempt in range(max_attempts):

        try:

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=90
            )

            if (
                response.status_code == 429
                or response.status_code >= 500
            ):

                if attempt == max_attempts - 1:
                    response.raise_for_status()

                retry_after = response.headers.get(
                    "Retry-After"
                )

                try:
                    wait_seconds = float(retry_after)
                except (TypeError, ValueError):
                    wait_seconds = min(
                        2 ** (attempt + 1), 30
                    )

                print(
                    f"Temporary API error {response.status_code}. "
                    f"Retrying in {wait_seconds} seconds "
                    f"({attempt + 1}/{max_attempts})..."
                )

                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            result = response.json()

            generated_text = (
                result["choices"][0]["message"]["content"]
            )

            if not generated_text:
                raise RuntimeError(
                    "The LLM returned an empty response."
                )

            return generated_text.strip()

        except requests.exceptions.RequestException:

            if attempt == max_attempts - 1:
                raise

            wait_seconds = min(
                2 ** (attempt + 1), 30
            )

            print(
                f"Connection error. Retrying in {wait_seconds} "
                f"seconds ({attempt + 1}/{max_attempts})..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "LLM request failed after all retry attempts."
    )


def answer_question(question, retriever, top_k=6, mode="hybrid"):
    """
    Retrieve evidence and generate an evidence-grounded answer.

    API errors are kept separate from genuine abstentions.
    """

    start = time.time()

    # 1. Retrieve relevant passages.
    evidence = retriever.search(
        question,
        top_k=top_k,
        mode=mode
    )

    # 2. Calculate the strongest dense retrieval score.
    best_dense = max(
        (
            e.get("dense_score", 0)
            for e in evidence
        ),
        default=0
    )

    threshold = float(
        os.getenv("ABSTAIN_THRESHOLD", "0.40")
    )

    # 3. Abstain if retrieval evidence is insufficient.
    if not evidence or best_dense < threshold:

        return {
            "answer": ABSTAIN_TEXT,
            "confidence": "INSUFFICIENT EVIDENCE",
            "status": "ABSTAINED",
            "evidence": [],
            "latency_seconds": round(
                time.time() - start, 3
            ),
            "llm_error": None
        }

    # 4. Generate an evidence-grounded answer.
    try:

        generated = call_llm(
            question,
            evidence
        )

    except Exception as exc:

        return {
            "answer": (
                "Answer generation failed because the LLM "
                "service was unavailable. Please retry."
            ),
            "confidence": "API ERROR",
            "status": "API_ERROR",
            "evidence": evidence,
            "latency_seconds": round(
                time.time() - start, 3
            ),
            "llm_error": str(exc)[:500]
        }

    # 5. Detect explicit LLM abstention.
    if generated.lstrip().startswith(ABSTAIN_MARKER):

        clean_answer = generated.lstrip()[
            len(ABSTAIN_MARKER):
        ].strip()

        if not clean_answer:
            clean_answer = ABSTAIN_TEXT

        return {
            "answer": clean_answer,
            "confidence": "INSUFFICIENT EVIDENCE",
            "status": "ABSTAINED",
            "evidence": evidence,
            "latency_seconds": round(
                time.time() - start, 3
            ),
            "llm_error": None
        }

    # 6. Calculate retrieval-based confidence.
    if best_dense >= 0.55:
        confidence = "HIGH"

    elif best_dense >= 0.35:
        confidence = "MEDIUM"

    else:
        confidence = "LOW"

    return {
        "answer": generated,
        "confidence": confidence,
        "status": "SUCCESS",
        "evidence": evidence,
        "latency_seconds": round(
            time.time() - start, 3
        ),
        "llm_error": None
    }
