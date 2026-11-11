# LexiSyncAI — Evidence-Grounded Legal Document Analyst

A beginner-friendly Retrieval-Augmented Generation (RAG) project for the Deep Learning final assignment.

> Educational demonstration only; not legal advice.

## 1. What It Does

LexiSyncAI is an AI-powered legal document analysis system that allows users to upload legal PDF documents, ask questions, and receive answers supported by document evidence.

The application provides the following features:

- Accepts 2–5 public legal PDFs through Streamlit.
- Extracts selectable PDF text and preserves page numbers using PyMuPDF.
- Splits text into heading-aware and clause-aware chunks.
- Retrieves relevant passages using BM25 keyword search.
- Generates dense embeddings using Sentence Transformers.
- Combines retrieval rankings using Reciprocal Rank Fusion (RRF).
- Optionally reranks candidates using a Cross-Encoder.
- Sends retrieved passages to an OpenAI-compatible LLM API.
- Displays source document, page number, and section information beside answers.
- Abstains when sufficient supporting evidence is unavailable.
- Supports evaluation of dense and hybrid retrieval methods.

The initial implementation uses two European legal documents:

- General Data Protection Regulation (GDPR).
- European Union Artificial Intelligence Act (EU AI Act).

## 2. Technologies Used

| Category | Technology |
|---|---|
| Programming language | Python |
| Deep Learning framework | PyTorch |
| PDF processing | PyMuPDF |
| Text embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Keyword retrieval | BM25 |
| Retrieval fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| LLM integration | Groq API / OpenAI-compatible API |
| User interface | Streamlit |
| Data handling | Pandas |
| Testing | Pytest |

## 3. Recommended Setup

Use Python 3.11 (recommended for package compatibility), rather than the newest Python release.

### Windows PowerShell

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## 4. Add Public Legal Documents

Place 2–5 public legal PDF documents inside:

```text
data/pdfs/
```

The Streamlit application also allows users to upload PDF documents directly.

Suitable public document sources include:

- EUR-Lex.
- SEC EDGAR.
- GOV.UK legislation.
- CourtListener.

Do not upload confidential or private client documents.

The initial project uses GDPR.pdf and AI_Act.pdf.

## 5. Configure Answer Generation

Create a `.env` file in the project root directory.

You can use `.env.example` as a template.

Example configuration:

```dotenv
LLM_API_KEY=your_groq_api_key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

USE_RERANKER=true
ABSTAIN_THRESHOLD=0.30
```

**Important:**
- Replace the example API key with your own valid key.
- Check your provider's current model availability.
- Never upload `.env` to GitHub.
- Keep your API key private.

If no API key is configured, the application can display retrieved evidence passages rather than presenting them as LLM-generated answers.

## 6. Project Structure

```text
LexiSyncAI/
│
├── app.py
├── evaluate.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── pdf_ingest.py
│   ├── retrieval.py
│   └── answer.py
│
├── data/
│   ├── pdfs/
│   │   ├── GDPR.pdf
│   │   └── AI_Act.pdf
│   │
│   ├── evaluation_questions.csv
│   └── evaluation_results.csv
│
└── tests/
    └── test_chunking.py
```

## 7. System Architecture

LexiSyncAI follows a Retrieval-Augmented Generation (RAG) architecture.

The main processing stages are:

1. **PDF ingestion:** Users upload legal PDF documents.
2. **Text extraction:** PyMuPDF extracts selectable text and preserves page information.
3. **Document chunking:** Extracted text is divided into meaningful chunks based on headings and legal sections.
4. **Dense embedding generation:** Sentence Transformers converts document chunks into vector representations.
5. **Keyword indexing:** BM25 creates a keyword-based retrieval index.
6. **Hybrid retrieval:** Dense retrieval and BM25 results are combined using Reciprocal Rank Fusion.
7. **Reranking:** A Cross-Encoder optionally reranks the retrieved candidates.
8. **Evidence selection:** The most relevant passages are selected for answer generation.
9. **LLM answer generation:** The LLM receives the question and retrieved evidence.
10. **Evidence-based response:** The application displays answers with document and page references.
11. **Abstention:** The system avoids generating unsupported answers when sufficient evidence is unavailable.

## 8. Running the Application

Activate your virtual environment first.

Run the following command:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

### How to use the application

1. Open LexiSyncAI in your browser.
2. Upload your legal PDF documents.
3. Allow the application to process the documents.
4. Enter a question related to the uploaded documents.
5. Select the retrieval method if the interface provides that option.
6. Review the generated answer.
7. Inspect the supporting evidence, page numbers, and section references.
8. Test an unanswerable question to observe the abstention behaviour.

The first run may take longer because the transformer models need to be downloaded.

CPU-based inference may also take additional time.

## 9. Evaluation Dataset

The project includes an evaluation dataset containing 15 questions.

The questions cover:

- Factual questions.
- Multi-part questions.
- Questions requiring legal document evidence.
- Unanswerable questions.

The dataset includes:

- 10 answerable questions.
- 5 unanswerable questions.

Both dense and hybrid retrieval methods are evaluated using the same questions.

### Evaluation questions

The questions are stored in:

```text
data/evaluation_questions.csv
```

### Run the evaluation

Execute:

```powershell
python evaluate.py
```

The evaluation results are saved to:

```text
data/evaluation_results.csv
```

The script compares dense and hybrid retrieval methods.

## 10. Evaluation Metrics

The following evaluation metrics are used:

### 1. Answer Correctness

Measures whether the generated answer correctly addresses the question.

### 2. Citation Precision

Measures whether the cited document pages and sections support the generated answer.

### 3. Evidence Recall@6

Measures whether the top six retrieved chunks contain the evidence required to answer the question.

Formula:

```text
Recall@6 =
Questions with gold evidence in top 6
/
Total answerable questions
```

### 4. Abstention Accuracy

Measures whether the system correctly answers answerable questions and abstains on unanswerable questions.

### 5. Average Latency

Measures the average time required to generate a response.

## 11. Initial Evaluation Results

The initial evaluation compared dense retrieval and hybrid retrieval using the same 15 questions.

| Evaluation Metric | Dense Retrieval | Hybrid Retrieval |
|---|---:|---:|
| Answer correctness | 100% | 100% |
| Citation precision | 100% | 100% |
| Evidence recall@6 | 51.7% | 51.7% |
| Correct abstention decisions | 100% | 100% |
| Average latency | 9.02 seconds | 11.15 seconds |

**Important evaluation notes:**

- These results are based on the current manual review of 15 questions per retrieval method.
- The answer correctness scores include the reviewed answerable questions and correct abstentions.
- Citation precision applies to answerable questions.
- Evidence recall values are preliminary manual estimates, not independently validated gold-standard measurements.
- The results should not be interpreted as proof of perfect performance on unseen legal questions.

### Result Analysis

Both retrieval methods successfully answered the reviewed answerable questions and correctly abstained on the unanswerable questions.

Dense retrieval achieved an average response time of approximately 9.02 seconds.

Hybrid retrieval achieved an average response time of approximately 11.15 seconds.

Therefore, hybrid retrieval was approximately 2.14 seconds slower on average.

The current evaluation did not show an improvement in estimated evidence recall from hybrid retrieval. Further testing with a larger dataset is required to determine whether hybrid retrieval provides measurable improvements.

## 12. Important Limitations

The current implementation has several limitations:

- OCR is not implemented in this starter version. Image-only pages are flagged.
- Table extraction is not fully implemented.
- Bounding-box citations are not implemented.
- Section detection uses heading patterns and is a baseline rather than a perfect legal parser.
- The abstention threshold is a starting value and requires calibration using validation questions.
- Similarity scores are not calibrated probabilities.
- Citation page and section metadata are attached to retrieved passages.
- Generated claims and citations must be manually checked for supporting evidence.
- The LLM can still generate incorrect or incomplete answers.
- The initial evaluation uses only two legal documents and 15 questions.

## 13. Future Improvements

Possible future improvements include:

- Implement OCR support for scanned legal documents.
- Improve legal clause and section detection.
- Expand the evaluation dataset.
- Improve evidence retrieval and recall.
- Add more European legal documents.
- Improve multi-hop question answering.
- Implement automated citation verification.
- Explore local LLM deployment.
- Add more comprehensive retrieval evaluation.
- Improve the Streamlit interface.

## 14. Running Tests

The project includes tests for document chunking.

Run:

```powershell
python -m pytest -q
```

The tests help verify that the document chunking functionality works as expected.

## 15. Suggested Demonstration Flow (5 Minutes)

The following demonstration structure can be used during the final assignment presentation.

| Time | Demonstration |
|---|---|
| 0:00–0:45 | Introduce the problem and project architecture. |
| 0:45–1:30 | Upload two public legal PDFs and show document processing. |
| 1:30–2:30 | Ask a factual question and inspect page-level evidence. |
| 2:30–3:30 | Ask a multi-hop question and explain evidence retrieval. |
| 3:30–4:15 | Ask an unanswerable question and demonstrate abstention. |
| 4:15–5:00 | Explain BM25, dense retrieval, hybrid retrieval, reranking, and evaluation results. |

## 16. Before Submission

- [ ] Verify that the application runs successfully.
- [ ] Test the application with the selected public legal PDFs.
- [ ] Prepare at least 15 evaluation questions.
- [ ] Include factual, multi-hop, and unanswerable questions.
- [ ] Manually verify generated answers and citations.
- [ ] Review evaluation metrics and results.
- [ ] Add screenshots of the working application to the final report.
- [ ] Include a system architecture diagram.
- [ ] Prepare an unseen public PDF for the demonstration.
- [ ] Test three unseen questions.
- [ ] Check that `.env` and API keys are excluded from GitHub.
- [ ] Include reproducible installation instructions.

---

## Project Information

**Project Name:** LexiSyncAI – Evidence-Grounded Legal Document Analyst

**Domain:** Deep Learning, Natural Language Processing, Information Retrieval, and Retrieval-Augmented Generation

**Initial Legal Documents:** GDPR and EU AI Act

**Interface:** Streamlit

**Purpose:** Educational demonstration of evidence-grounded legal document question answering.

> This application is intended for educational and research purposes only. It does not provide legal advice.