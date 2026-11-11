
import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.pdf_ingest import process_pdf
from src.retrieval import HybridRetriever
from src.answer import answer_question


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="LexiSyncAI",
    page_icon="⚖️",
    layout="wide"
)

ABSTAIN_TEXT = (
    "The documents do not contain enough evidence to answer this."
)


# ============================================================
# 2. CITATION VALIDATION
# ============================================================

def validate_citations(answer, evidence):
    """
    Validate citation references against retrieved evidence.

    Supports both:
        [Source 1, GDPR.pdf, p. 71, Article 83(4)]
        【Source 1, GDPR.pdf, p. 71, Article 83(4)】

    This checks reference consistency, not legal correctness.
    """

    citation_pattern = re.compile(
        r"(?:\[|【)\s*source\s+(\d+)([^\]】]*)(?:\]|】)",
        re.IGNORECASE
    )

    citations = citation_pattern.findall(answer or "")

    total_citations = len(citations)
    reference_matches = 0
    reference_errors = 0

    details = []

    for source_number_text, citation_details in citations:

        source_number = int(source_number_text)

        # Source numbers in the answer start at 1.
        if source_number < 1 or source_number > len(evidence):

            reference_errors += 1

            details.append({
                "citation": f"Source {source_number}",
                "status": "INVALID",
                "reason": "Source number does not exist."
            })

            continue

        source = evidence[source_number - 1]

        document = str(source.get("document", ""))
        page = str(source.get("page", ""))
        section = str(source.get("section", ""))
        text_content = str(source.get("text", ""))

        errors = []

        # Check document name.
        if document.lower() not in citation_details.lower():
            errors.append("Document name mismatch.")

        # Check page number.
        page_match = re.search(
            r"\b(?:p\.?|page)\s*(\d+)\b",
            citation_details,
            re.IGNORECASE
        )

        if page_match:
            cited_page = page_match.group(1)

            if cited_page != page:
                errors.append("Page number mismatch.")
        else:
            errors.append("Page number missing.")

        # Check article number, allowing paragraph references.
        article_match = re.search(
            r"\barticle\s+(\d+[a-z]?)",
            citation_details,
            re.IGNORECASE
        )

        if article_match:

            cited_article = article_match.group(1).lower()

            actual_article_match = re.search(
                r"\barticle\s+(\d+[a-z]?)",
                section,
                re.IGNORECASE
            )

            article_matches = (
                actual_article_match is not None
                and actual_article_match.group(1).lower()
                == cited_article
            )

            if not article_matches:

                # Some chunks may have article references in their text
                # even when section metadata is unavailable.
                article_matches = bool(
                    re.search(
                        rf"\barticle\s+{re.escape(cited_article)}\b",
                        text_content,
                        re.IGNORECASE
                    )
                )

            if not article_matches:
                errors.append("Article reference mismatch.")

        if errors:

            reference_errors += 1

            details.append({
                "citation": f"Source {source_number}",
                "status": "INVALID",
                "reason": " ".join(errors)
            })

        else:

            reference_matches += 1

            details.append({
                "citation": f"Source {source_number}",
                "status": "MATCH",
                "reason": (
                    f"{document}, page {page}, {section}"
                )
            })

    return {
        "total_citations": total_citations,
        "reference_matches": reference_matches,
        "reference_errors": reference_errors,
        "details": details
    }


# ============================================================
# 3. PDF PROCESSING
# ============================================================

def process_uploaded_pdfs(uploaded_files):

    all_chunks = []
    document_names = []

    progress = st.progress(0)
    status = st.empty()

    for index, uploaded_file in enumerate(uploaded_files):

        filename = Path(uploaded_file.name).name

        status.write(f"Processing {filename}...")

        temp_path = None

        try:

            # Save uploaded PDF temporarily.
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            # Extract page-level evidence chunks.
            chunks, metadata = process_pdf(
                temp_path,
                filename
            )

            if not chunks:
                st.warning(
                    f"No readable text found in {filename}."
                )
                continue

            all_chunks.extend(chunks)
            document_names.append(filename)

            st.success(
                f"{filename}: {len(chunks)} evidence chunks extracted."
            )

        except Exception as exc:

            st.error(
                f"Could not process {filename}: {exc}"
            )

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        progress.progress(
            (index + 1) / len(uploaded_files)
        )

    progress.empty()
    status.empty()

    return all_chunks, document_names


# ============================================================
# 4. HEADER
# ============================================================

st.title("⚖️ LexiSyncAI")

st.subheader(
    "Evidence-Grounded Legal Document Analyst"
)

st.caption(
    "Educational demonstration only. This tool is not legal advice."
)

st.divider()


# ============================================================
# 5. SESSION STATE
# ============================================================

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "document_names" not in st.session_state:
    st.session_state.document_names = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# 6. PDF UPLOAD SECTION
# ============================================================

st.header("1 · Upload legal PDFs")

st.write("Choose 2–5 public PDF documents.")

uploaded_files = st.file_uploader(
    "Upload your legal documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload between 2 and 5 public legal PDF documents."
)

if uploaded_files:

    if len(uploaded_files) < 2 or len(uploaded_files) > 5:

        st.warning(
            "Please upload between 2 and 5 PDF documents."
        )

    else:

        st.write("Selected documents:")

        for uploaded_file in uploaded_files:
            st.write(f"📄 {uploaded_file.name}")

        if st.button(
            "Process and index documents",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Extracting documents and building retrieval index..."
            ):

                chunks, document_names = process_uploaded_pdfs(
                    uploaded_files
                )

                if not chunks:

                    st.error(
                        "No readable evidence was extracted."
                    )

                else:

                    try:

                        retriever = HybridRetriever(chunks)

                        st.session_state.retriever = retriever

                        st.session_state.document_names = (
                            document_names
                        )

                        st.session_state.total_chunks = len(chunks)

                        st.session_state.history = []

                        st.success(
                            "Documents successfully indexed!"
                        )

                    except Exception as exc:

                        st.error(
                            f"Index creation failed: {exc}"
                        )

                        st.session_state.retriever = None


# ============================================================
# 7. INDEXED DOCUMENTS
# ============================================================

if st.session_state.retriever is not None:

    st.subheader("Indexed documents")

    for document in st.session_state.document_names:
        st.write(f"📄 {document}")

    st.metric(
        "Total evidence chunks",
        st.session_state.total_chunks
    )

else:

    st.info(
        "Upload and process 2–5 PDFs to start asking questions."
    )


st.divider()


# ============================================================
# 8. QUESTION AND ANSWER SECTION
# ============================================================

st.header("2 · Ask a question")

st.write(
    "Ask a question about the uploaded legal documents."
)

if st.session_state.retriever is not None:

    question = st.text_area(
        "Your question",
        placeholder=(
            "Example: What is the maximum administrative "
            "fine under GDPR Article 83(5)?"
        ),
        height=100
    )

    col1, col2 = st.columns(2)

    with col1:

        top_k = st.slider(
            "Evidence passages to retrieve",
            min_value=3,
            max_value=12,
            value=6
        )

    with col2:

        retrieval_mode = st.selectbox(
            "Retrieval method",
            options=["hybrid", "dense", "bm25"],
            index=0
        )

    ask_button = st.button(
        "Generate answer",
        type="primary",
        use_container_width=True
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "Searching evidence and generating your answer..."
            ):

                try:

                    result = answer_question(
                        question=question.strip(),
                        retriever=st.session_state.retriever,
                        top_k=top_k,
                        mode=retrieval_mode
                    )

                    answer = result.get(
                        "answer",
                        ABSTAIN_TEXT
                    )

                    evidence = result.get(
                        "evidence",
                        []
                    )

                    confidence = result.get(
                        "confidence",
                        "UNKNOWN"
                    )

                    latency = result.get(
                        "latency_seconds",
                        0
                    )

                    # Support newer answer.py status field
                    # without requiring it in older versions.
                    answer_status = result.get(
                        "status",
                        "SUCCESS"
                    )

                    if answer_status == "API_ERROR":
                        st.warning(
                            "LLM API failed. Review the returned message "
                            "before relying on this result."
                        )

                    elif (
                        confidence == "INSUFFICIENT EVIDENCE"
                        or answer.strip() == ABSTAIN_TEXT
                    ):
                        answer_status = "ABSTAINED"

                    else:
                        answer_status = "SUCCESS"

                    validation = validate_citations(
                        answer,
                        evidence
                    )

                    st.session_state.history.append({
                        "question": question,
                        "answer": answer,
                        "confidence": confidence,
                        "evidence": evidence,
                        "validation": validation,
                        "latency": latency,
                        "status": answer_status
                    })

                except Exception as exc:

                    st.error(
                        f"An error occurred while answering: {exc}"
                    )


# ============================================================
# 9. DISPLAY LATEST ANSWER
# ============================================================

if st.session_state.history:

    latest = st.session_state.history[-1]

    st.divider()

    st.header("Answer")

    st.markdown(latest["answer"])

    confidence = latest["confidence"]

    if confidence == "HIGH":
        st.success("Confidence: HIGH")

    elif confidence == "MEDIUM":
        st.info("Confidence: MEDIUM")

    elif confidence == "LOW":
        st.warning("Confidence: LOW")

    else:
        st.warning(
            f"Confidence: {confidence}"
        )

    # --------------------------------------------------------
    # CITATION VALIDATION
    # --------------------------------------------------------

    st.subheader("Citation reference validation")

    validation = latest["validation"]

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Total citations",
        validation["total_citations"]
    )

    metric2.metric(
        "Reference matches",
        validation["reference_matches"]
    )

    metric3.metric(
        "Reference errors",
        validation["reference_errors"]
    )

    if validation["total_citations"] == 0:

        st.warning(
            "No recognised source citations were found in the answer."
        )

    elif validation["reference_errors"] == 0:

        st.success(
            "All citation references match the retrieved evidence."
        )

    else:

        st.warning(
            "Some citation references do not match the retrieved evidence."
        )

    with st.expander("View citation validation details"):

        if validation["details"]:

            for detail in validation["details"]:

                if detail["status"] == "MATCH":
                    st.success(
                        f'{detail["citation"]}: {detail["reason"]}'
                    )
                else:
                    st.error(
                        f'{detail["citation"]}: {detail["reason"]}'
                    )

        else:

            st.write("No citation references to validate.")

    st.info(
        "Important: This validator checks whether citation references "
        "match retrieved source numbers, document names, pages and "
        "articles. It does not establish whether the cited passage "
        "legally supports the claim."
    )

    # --------------------------------------------------------
    # EVIDENCE DISPLAY
    # --------------------------------------------------------

    st.subheader("Evidence used")

    evidence = latest["evidence"]

    if evidence:

        for index, item in enumerate(evidence, start=1):

            document = item.get(
                "document",
                "Unknown document"
            )

            page = item.get(
                "page",
                "Unknown"
            )

            section = item.get(
                "section",
                "Unknown section"
            )

            with st.expander(
                f"Source {index} · {document} · "
                f"Page {page} · {section}"
            ):

                st.write(item.get("text", ""))

                st.caption(
                    f"Chunk ID: {item.get('chunk_id', 'Not available')}"
                )

                if "dense_score" in item:
                    st.caption(
                        f"Dense score: {item['dense_score']:.4f}"
                    )

                if "bm25_score" in item:
                    st.caption(
                        f"BM25 score: {item['bm25_score']:.4f}"
                    )

    else:

        st.warning("No evidence passages were retrieved.")

    st.caption(
        f"Latency: {latest['latency']:.2f} seconds"
    )


# ============================================================
# 10. PREVIOUS QUESTIONS
# ============================================================

if len(st.session_state.history) > 1:

    st.divider()

    with st.expander("Previous questions"):

        for index, item in enumerate(
            reversed(st.session_state.history[:-1]),
            start=1
        ):

            st.markdown(
                f"**Question {index}:** {item['question']}"
            )

            st.write(item["answer"])

            st.caption(
                f"Confidence: {item['confidence']} | "
                f"Latency: {item['latency']:.2f} seconds"
            )

            st.divider()


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "LexiSyncAI · Deep Learning final assignment · "
    "Built for learning and demonstration"
)
