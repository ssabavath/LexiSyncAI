
"""Hybrid legal retrieval with paragraph-aware and multi-document support."""

import os
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


def tokenize(text):
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"[a-z0-9§.]+", str(text).lower())


def is_comparison_question(query):
    """Detect comparison wording."""
    terms = (
        "compare", "comparison", "differ", "difference", "differences",
        "versus", " vs ", "contrast",
    )
    q = f" {query.lower()} "
    return any(term in q for term in terms)


class HybridRetriever:
    """Retrieve legal evidence using dense, BM25 and hybrid search."""

    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No chunks supplied.")

        self.chunks = chunks
        self.document_groups = {}

        for index, chunk in enumerate(chunks):
            document = chunk["document"]
            self.document_groups.setdefault(document, []).append(index)

        tokenized_corpus = [
            tokenize(chunk["text"]) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        embedding_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Loading embedding model:", embedding_name)

        self.embedder = SentenceTransformer(
            embedding_name,
            device="cpu"
        )

        self.embeddings = self.embedder.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        self.reranker = None

        if os.getenv("USE_RERANKER", "true").lower() == "true":
            try:
                reranker_name = os.getenv(
                    "RERANKER_MODEL",
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )

                print("Loading reranker:", reranker_name)

                self.reranker = CrossEncoder(
                    reranker_name,
                    device="cpu"
                )

            except Exception as exc:
                print(
                    "Reranker unavailable; continuing without it:",
                    exc
                )

    def _is_gdpr_document(self, name):
        return "gdpr" in str(name).lower()

    def _is_ai_act_document(self, name):
        name = str(name).lower()

        return (
            "ai_act" in name
            or "ai act" in name
            or "artificial intelligence act" in name
        )

    def _rank_candidates(
        self,
        bm25_scores,
        dense_scores,
        candidate_indexes,
        mode,
        candidate_k
    ):
        """Rank candidates using dense, BM25 or reciprocal rank fusion."""

        bm_order = sorted(
            candidate_indexes,
            key=lambda idx: bm25_scores[idx],
            reverse=True
        )

        dense_order = sorted(
            candidate_indexes,
            key=lambda idx: dense_scores[idx],
            reverse=True
        )

        if mode == "dense":
            return dense_order[:candidate_k]

        if mode == "bm25":
            return bm_order[:candidate_k]

        if mode != "hybrid":
            raise ValueError(
                "mode must be 'dense', 'bm25', or 'hybrid'"
            )

        rrf = {}

        for order in (
            bm_order[:candidate_k],
            dense_order[:candidate_k]
        ):
            for rank, idx in enumerate(order, start=1):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)

        return sorted(
            rrf,
            key=rrf.get,
            reverse=True
        )[:candidate_k]

    def _get_document_hint(self, query):
        """Identify the regulation explicitly or contextually indicated."""

        q = query.lower()

        mentions_gdpr = "gdpr" in q

        mentions_ai_act = (
            "ai act" in q
            or "artificial intelligence act" in q
        )

        # Explicit document names take priority.
        if mentions_gdpr and not mentions_ai_act:
            return "gdpr"

        if mentions_ai_act and not mentions_gdpr:
            return "ai_act"

        if mentions_gdpr and mentions_ai_act:
            return None

        # Contextual GDPR indicators.
        gdpr_indicators = (
            "fine",
           "fines",
          "administrative fine",
         "administrative fines",
           "worldwide annual turnover",
          "data subject",
          "supervisory authority",
           "controller or processor",
          "personal data",
         "processing principles",
          "consent",
        )
        mentions_article_83_or_84 = bool(
            re.search(r"\barticle\s+(?:83|84)\b", q)
        )

        has_gdpr_document = any(
            self._is_gdpr_document(name)
            for name in self.document_groups
        )

        if (
            mentions_article_83_or_84
            and any(term in q for term in gdpr_indicators)
            and has_gdpr_document
        ):
            return "gdpr"

        # AI Act-specific vocabulary.
        ai_indicators = (
            "high-risk ai",
            "high risk ai",
            "ce marking",
            "market surveillance",
            "ai system",
            "union ai testing support structures",
        )

        has_ai_document = any(
            self._is_ai_act_document(name)
            for name in self.document_groups
        )

        if (
            any(term in q for term in ai_indicators)
            and has_ai_document
        ):
            return "ai_act"

        return None

    def _is_cross_document_question(self, query):
        """Only activate multi-document mode for clear cross-document intent."""

        q = query.lower()

        explicit_cross_document_phrases = (
            "both documents",
            "both regulations",
            "across documents",
            "across both",
            "gdpr and ai act",
            "ai act and gdpr",
            "between the gdpr and the ai act",
            "between gdpr and ai act",
            "compare the gdpr with the ai act",
            "compare gdpr and ai act",
        )

        return any(
            phrase in q
            for phrase in explicit_cross_document_phrases
        )

    def _requested_articles(self, query):
        matches = re.findall(
            r"\barticle\s+(\d+[a-z]?)",
            query,
            flags=re.IGNORECASE
        )

        return list(dict.fromkeys(
            f"article {number.lower()}"
            for number in matches
        ))

    def _find_article_chunks(self, query):
        """Find requested article chunks, with GDPR fines context."""

        query_lower = query.lower()

        # Find ALL article numbers mentioned, not just the first.
        matches = re.findall(
            r"\barticle\s+(\d+[a-z]?)\b",
            query_lower
        )

        requested_sections = {
            f"article {number.lower()}"
            for number in matches
        }

        if not requested_sections:
            return []

        # Infer GDPR for questions about its administrative fines.
        gdpr_fines_question = any(term in query_lower for term in [
            "administrative fine",
            "administrative fines",
            "worldwide annual turnover",
            "data subject",
            "supervisory authority",
            "processing principles",
            "paragraphs 4 and 5",
            "paragraph 4 and 5",
        ])

        document_hint = None

        if "gdpr" in query_lower or gdpr_fines_question:
            document_hint = "gdpr"

        elif (
            "ai act" in query_lower
            or "artificial intelligence act" in query_lower
        ):
            document_hint = "ai_act"

        article_indexes = []

        for index, chunk in enumerate(self.chunks):
            section = str(chunk.get("section", "")).lower()
            document = str(chunk.get("document", "")).lower()

            if section not in requested_sections:
                continue

            if document_hint == "gdpr" and not self._is_gdpr_document(document):
                continue

            if document_hint == "ai_act" and not self._is_ai_act_document(document):
                continue

            article_indexes.append(index)

        return article_indexes

    def _extract_article_paragraph_references(self, query):
        """Return article/paragraph pairs explicitly or contextually referenced."""

        refs = []

        for article, paragraph in re.findall(
            r"\barticle\s+(\d+[a-z]?)\s*\(\s*(\d+)\s*\)",
            query,
            flags=re.IGNORECASE
        ):
            refs.append((article.lower(), paragraph))

        # Handle wording such as:
        # Article 83(2) ... paragraphs 4 and 5.
        article_match = re.search(
            r"\barticle\s+(\d+[a-z]?)\b",
            query,
            flags=re.IGNORECASE
        )

        if article_match:
            article_number = article_match.group(1).lower()

            contextual_paragraphs = re.search(
                r"\bparagraphs?\s+(\d+)(?:\s*(?:,|and|&)\s*(\d+))*",
                query,
                flags=re.IGNORECASE
            )

            if contextual_paragraphs:
                phrase = contextual_paragraphs.group(0)

                for paragraph in re.findall(r"\d+", phrase):
                    refs.append((article_number, paragraph))

        return list(dict.fromkeys(refs))

    def _find_article_paragraph_chunks(self, query):
        """Find chunks containing requested paragraph markers and continuations."""

        references = self._extract_article_paragraph_references(query)

        if not references:
            return []

        document_hint = self._get_document_hint(query)
        selected_indexes = []

        for article_number, paragraph_number in references:
            target_section = f"article {article_number}"
            article_indexes = []

            for index, chunk in enumerate(self.chunks):
                section = str(chunk.get("section", "")).lower()
                document = str(chunk.get("document", "")).lower()

                if section != target_section:
                    continue

                if (
                    document_hint == "gdpr"
                    and not self._is_gdpr_document(document)
                ):
                    continue

                if (
                    document_hint == "ai_act"
                    and not self._is_ai_act_document(document)
                ):
                    continue

                article_indexes.append(index)

            paragraph_pattern = re.compile(
                rf"(?m)^[ \t]*{re.escape(paragraph_number)}\.[ \t]*(?=\S|$)"
            )

            found = False

            for position, index in enumerate(article_indexes):
                text = self.chunks[index].get("text", "")

                if paragraph_pattern.search(text):
                    selected_indexes.append(index)

                    # Include the next chunk if the paragraph continues.
                    if position + 1 < len(article_indexes):
                        selected_indexes.append(
                            article_indexes[position + 1]
                        )

                    found = True
                    break

            if not found:
                # Fallback to paragraph metadata when available.
                for index in article_indexes:
                    paragraph = str(
                        self.chunks[index].get("paragraph", "")
                    )

                    if paragraph == paragraph_number:
                        selected_indexes.append(index)
                        break

        return sorted(set(selected_indexes))

    def _find_concept_article_chunks(self, query):
        """Add candidates for recognised legal concepts."""

        q = query.lower()

        requested = {
            "gdpr": set(),
            "ai_act": set()
        }

        # GDPR Article 5: processing principles.
        if any(term in q for term in (
            "principle",
            "principles",
            "processing personal data",
            "personal data processing",
        )):
            requested["gdpr"].add("article 5")

        # GDPR Article 25: data protection by design.
        if any(term in q for term in (
            "data protection by design",
            "privacy by design",
            "by default",
        )):
            requested["gdpr"].add("article 25")

        # GDPR Article 83: administrative fines.
        fine_terms = (
            "fine",
            "fines",
            "turnover",
            "annual revenue",
            "annual worldwide",
            "4%",
            "4 percent",
            "2%",
            "2 percent",
            "administrative penalty",
            "administrative penalties",
            "administrative fine",
            "administrative fines",
        )

        if any(term in q for term in fine_terms):
            requested["gdpr"].add("article 83")

        # GDPR Article 84: other penalties.
        if any(term in q for term in (
            "other penalties",
            "other applicable penalties",
            "article 84",
            "penalties for infringements",
        )):
            requested["gdpr"].add("article 84")

        # AI Act Article 9: risk management.
        if any(term in q for term in (
            "high-risk",
            "high risk",
            "risk management",
        )):
            requested["ai_act"].add("article 9")

        # AI Act Article 10: data governance.
        if any(term in q for term in (
            "high-risk",
            "high risk",
            "data governance",
            "training data",
            "validation data",
        )):
            requested["ai_act"].add("article 10")

        results = []

        for index, chunk in enumerate(self.chunks):
            section = str(chunk.get("section", "")).lower()
            document = str(chunk.get("document", ""))

            if (
                self._is_gdpr_document(document)
                and section in requested["gdpr"]
            ):
                results.append(index)

            elif (
                self._is_ai_act_document(document)
                and section in requested["ai_act"]
            ):
                results.append(index)

        return results

    def _rerank(self, query, candidates):
        """Rerank candidate passages with a cross-encoder."""

        if not self.reranker or not candidates:
            return candidates

        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
            candidate["score"] = float(score)

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return candidates

    def search(
        self,
        query,
        top_k=6,
        candidate_k=30,
        mode="hybrid"
    ):
        """Retrieve and rerank relevant legal evidence."""

        if top_k < 1 or candidate_k < 1:
            raise ValueError(
                "top_k and candidate_k must be positive."
            )

        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError(
                "mode must be 'dense', 'bm25', or 'hybrid'."
            )

        # 1. Calculate BM25 and dense scores.
        tokens = tokenize(query)

        bm25_scores = np.asarray(
            self.bm25.get_scores(tokens),
            dtype=float
        )

        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        dense_scores = self.embeddings @ query_embedding

        # 2. Detect document names and contextual document scope.
        q = query.lower()

        mentions_gdpr = "gdpr" in q

        mentions_ai_act = (
            "ai act" in q
            or "artificial intelligence act" in q
        )

        available_gdpr = any(
            self._is_gdpr_document(name)
            for name in self.document_groups
        )

        available_ai_act = any(
            self._is_ai_act_document(name)
            for name in self.document_groups
        )

        named_multi_document = (
            mentions_gdpr
            and mentions_ai_act
            and available_gdpr
            and available_ai_act
        )

        document_hint = self._get_document_hint(query)

        comparison_mode = (
            self._is_cross_document_question(query)
            and document_hint is None
            and not mentions_gdpr
            and not mentions_ai_act
            and available_gdpr
            and available_ai_act
        )

        multi_document_mode = (
            named_multi_document or comparison_mode
        )

        if named_multi_document:
            print("Multi-document retrieval enabled: GDPR + AI Act")

        elif comparison_mode:
            print(
                "Explicit cross-document retrieval enabled:",
                list(self.document_groups.keys())
            )

        # 3. Find explicit article, paragraph and concept candidates.
        paragraph_indexes = self._find_article_paragraph_chunks(query)

        explicit_indexes = self._find_article_chunks(query)

        concept_indexes = self._find_concept_article_chunks(query)

        priority_indexes = list(dict.fromkeys(
            paragraph_indexes
            + explicit_indexes
            + concept_indexes
        ))

        if paragraph_indexes:
            print(
                "Paragraph-specific candidates added:",
                len(paragraph_indexes)
            )

        elif priority_indexes:
            print(
                "Article/concept candidates added:",
                len(priority_indexes)
            )

        # 4. Restrict search to the relevant document when possible.
        if document_hint == "gdpr":
            scope_indexes = [
                index
                for name, indexes in self.document_groups.items()
                if self._is_gdpr_document(name)
                for index in indexes
            ]

            print("Document scope: GDPR")

        elif document_hint == "ai_act":
            scope_indexes = [
                index
                for name, indexes in self.document_groups.items()
                if self._is_ai_act_document(name)
                for index in indexes
            ]

            print("Document scope: AI Act")

        else:
            scope_indexes = list(range(len(self.chunks)))

        # 5. Retrieve ordinary candidates.
        candidate_ids = []

        if multi_document_mode:
            for document_name, indexes in self.document_groups.items():
                document_candidates = self._rank_candidates(
                    bm25_scores,
                    dense_scores,
                    indexes,
                    mode,
                    candidate_k
                )

                candidate_ids.extend(document_candidates)

        else:
            candidate_ids = self._rank_candidates(
                bm25_scores,
                dense_scores,
                scope_indexes,
                mode,
                candidate_k
            )

        # Only add priority candidates within the applicable scope.
        allowed_indexes = set(scope_indexes)

        candidate_ids = list(dict.fromkeys(
            candidate_ids
            + [
                index
                for index in priority_indexes
                if index in allowed_indexes
            ]
        ))

        # 6. Build candidate objects.
        candidates = []

        for idx in candidate_ids:
            item = dict(self.chunks[idx])

            item["bm25_score"] = float(bm25_scores[idx])
            item["dense_score"] = float(dense_scores[idx])
            item["score"] = float(dense_scores[idx])

            candidates.append(item)

        # 7. Cross-encoder reranking.
        candidates = self._rerank(query, candidates)

        candidate_lookup = {
            item["chunk_id"]: item
            for item in candidates
        }

        # 8. Select paragraph-specific evidence first.
        selected = []
        selected_ids = set()

        def add_index(index):
            if index not in allowed_indexes:
                return

            chunk_id = self.chunks[index]["chunk_id"]
            item = candidate_lookup.get(chunk_id)

            if item is not None and chunk_id not in selected_ids:
                selected.append(item)
                selected_ids.add(chunk_id)

        for index in paragraph_indexes:
            if len(selected) >= top_k:
                break

            add_index(index)

        # 9. Ensure explicitly requested articles are represented.
        requested_articles = set(
            self._requested_articles(query)
        )

        if requested_articles:
            for article in sorted(requested_articles):
                if len(selected) >= top_k:
                    break

                article_candidates = [
                    item
                    for item in candidates
                    if str(item.get("section", "")).lower() == article
                    and item["chunk_id"] not in selected_ids
                ]

                if article_candidates:
                    item = article_candidates[0]
                    selected.append(item)
                    selected_ids.add(item["chunk_id"])

        # 10. For cross-document questions, include both documents.
        if multi_document_mode:
            for document_name in self.document_groups:
                if len(selected) >= top_k:
                    break

                if not any(
                    item.get("document") == document_name
                    for item in selected
                ):
                    document_results = [
                        item
                        for item in candidates
                        if item.get("document") == document_name
                        and item["chunk_id"] not in selected_ids
                    ]

                    if document_results:
                        item = document_results[0]
                        selected.append(item)
                        selected_ids.add(item["chunk_id"])

        # 11. Fill remaining positions using reranked candidates.
        for item in candidates:
            if len(selected) >= top_k:
                break

            if item["chunk_id"] not in selected_ids:
                selected.append(item)
                selected_ids.add(item["chunk_id"])

        return selected[:top_k]
