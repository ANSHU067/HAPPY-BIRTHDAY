"""Prompt builder for RAG applications.

This module handles prompt engineering and context formatting for RAG pipelines.
"""

from typing import Dict, List, Optional, Tuple

from app.core.rag_config import get_rag_config


class PromptBuilder:
    """Builds prompts for RAG applications."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_context_length: Optional[int] = None,
    ):
        """Initialize the prompt builder.

        Args:
            system_prompt: Default system prompt
            max_context_length: Maximum context length in characters
        """
        rag_config = get_rag_config()

        self.max_context_length = max_context_length or rag_config.max_context_length
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a helpful AI assistant that answers questions based on the provided context.

Guidelines:
1. Answer questions using ONLY the information from the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific parts of the context when answering
4. Be concise and direct in your responses
5. If you're unsure, express that uncertainty rather than guessing
"""

    def build_rag_prompt(
        self,
        question: str,
        context_documents: List[Tuple[str, float, Dict]],
        include_metadata: bool = True,
        include_scores: bool = False,
    ) -> str:
        """Build a RAG prompt from question and context.

        Args:
            question: User's question
            context_documents: List of (text, score, metadata) tuples
            include_metadata: Whether to include document metadata
            include_scores: Whether to include relevance scores

        Returns:
            Formatted prompt string
        """
        # Build context section
        context_parts = []
        current_length = 0

        for i, (text, score, metadata) in enumerate(context_documents, 1):
            # Build document header
            header_parts = [f"Document {i}"]
            if include_scores:
                header_parts.append(f"(Relevance: {score:.2f})")
            if include_metadata and metadata:
                meta_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                header_parts.append(f"[{meta_str}]")

            header = " ".join(header_parts)
            doc_section = f"\n{header}:\n{text}\n"

            # Check length limit
            if current_length + len(doc_section) > self.max_context_length:
                break

            context_parts.append(doc_section)
            current_length += len(doc_section)

        context = "".join(context_parts)

        # Build full prompt
        prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

        return prompt

    def build_rag_messages(
        self,
        question: str,
        context_documents: List[Tuple[str, float, Dict]],
        include_metadata: bool = True,
        include_scores: bool = False,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build messages for chat-based models.

        Args:
            question: User's question
            context_documents: List of (text, score, metadata) tuples
            include_metadata: Whether to include document metadata
            include_scores: Whether to include relevance scores
            system_prompt: Override system prompt

        Returns:
            List of message dictionaries
        """
        # Build context
        context_parts = []
        current_length = 0

        for i, (text, score, metadata) in enumerate(context_documents, 1):
            header_parts = [f"Document {i}"]
            if include_scores:
                header_parts.append(f"(Relevance: {score:.2f})")
            if include_metadata and metadata:
                meta_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()])
                header_parts.append(f"[{meta_str}]")

            header = " ".join(header_parts)
            doc_section = f"\n{header}:\n{text}\n"

            if current_length + len(doc_section) > self.max_context_length:
                break

            context_parts.append(doc_section)
            current_length += len(doc_section)

        context = "".join(context_parts)

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {
                "role": "user",
                "content": f"""Context:
{context}

Question: {question}""",
            },
        ]

        return messages

    def build_conversational_prompt(
        self,
        question: str,
        context_documents: List[Tuple[str, float, Dict]],
        conversation_history: List[Dict[str, str]],
        max_history: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Build prompt with conversation history.

        Args:
            question: Current question
            context_documents: Retrieved context
            conversation_history: Previous messages
            max_history: Maximum number of history messages to include

        Returns:
            List of message dictionaries
        """
        rag_config = get_rag_config()
        max_hist = max_history or rag_config.max_conversation_history

        # Build context
        context_parts = []
        current_length = 0

        for i, (text, score, metadata) in enumerate(context_documents, 1):
            doc_section = f"\nDocument {i}:\n{text}\n"
            if current_length + len(doc_section) > self.max_context_length:
                break
            context_parts.append(doc_section)
            current_length += len(doc_section)

        context = "".join(context_parts)

        # Build messages with history
        messages = [
            {
                "role": "system",
                "content": self.system_prompt + f"\n\nRelevant context:\n{context}",
            }
        ]

        # Add recent conversation history
        recent_history = (
            conversation_history[-max_hist:] if conversation_history else []
        )
        messages.extend(recent_history)

        # Add current question
        messages.append({"role": "user", "content": question})

        return messages

    def build_citation_prompt(
        self, question: str, context_documents: List[Tuple[str, float, Dict]]
    ) -> str:
        """Build prompt that encourages citation.

        Args:
            question: User's question
            context_documents: Retrieved context with metadata

        Returns:
            Formatted prompt with citation instructions
        """
        # Number documents for citation
        context_parts = []
        for i, (text, score, metadata) in enumerate(context_documents, 1):
            source = metadata.get("source", f"Source {i}")
            context_parts.append(f"[{i}] {source}:\n{text}\n")

        context = "\n".join(context_parts)

        prompt = f"""{self.system_prompt}

When answering, cite your sources using [1], [2], etc.

Context:
{context}

Question: {question}

Answer (with citations):"""

        return prompt


class PromptTemplates:
    """Collection of prompt templates for different use cases."""

    @staticmethod
    def qa_template() -> str:
        """Question answering template."""
        return """Answer the following question based on the context provided.

Context:
{context}

Question: {question}

Answer:"""

    @staticmethod
    def summarization_template() -> str:
        """Document summarization template."""
        return """Summarize the following documents concisely.

Documents:
{context}

Summary:"""

    @staticmethod
    def fact_checking_template() -> str:
        """Fact checking template."""
        return """Verify the following claim against the provided context.

Context:
{context}

Claim: {question}

Verification (state if the claim is supported, contradicted, or not mentioned):"""

    @staticmethod
    def comparison_template() -> str:
        """Comparison template."""
        return """Compare and contrast the information from multiple sources.

Sources:
{context}

Question: {question}

Comparison:"""


if __name__ == "__main__":
    # Example usage
    builder = PromptBuilder()

    question = "What is RAG and how does it work?"
    context_docs = [
        (
            "RAG (Retrieval Augmented Generation) combines retrieval with generation. It retrieves relevant documents and uses them to generate informed responses.",
            0.92,
            {"source": "rag_overview.pdf", "page": 1},
        ),
        (
            "The RAG process involves: 1) embedding the query, 2) retrieving relevant documents, 3) reranking results, 4) generating a response using the context.",
            0.87,
            {"source": "rag_tutorial.md", "section": "Process"},
        ),
        (
            "Vector databases are commonly used in RAG systems to store and retrieve document embeddings efficiently.",
            0.75,
            {"source": "vector_db_guide.pdf", "page": 5},
        ),
    ]

    # Standard RAG prompt
    print("=== Standard RAG Prompt ===")
    prompt = builder.build_rag_prompt(question, context_docs, include_scores=True)
    print(prompt)

    # Chat messages format
    print("\n=== Chat Messages Format ===")
    messages = builder.build_rag_messages(question, context_docs, include_scores=True)
    for msg in messages:
        print(f"\n{msg['role'].upper()}:")
        print(msg["content"])

    # Citation prompt
    print("\n=== Citation Prompt ===")
    citation_prompt = builder.build_citation_prompt(question, context_docs)
    print(citation_prompt)
