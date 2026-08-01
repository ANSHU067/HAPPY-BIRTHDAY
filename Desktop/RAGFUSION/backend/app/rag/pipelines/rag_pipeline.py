"""RAG Pipeline using LangGraph for orchestration.

This module implements the complete RAG workflow using LangGraph's state management.
"""

import os
from typing import Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.core.rag_config import get_rag_config
from app.rag.embeddings.embedding_manager import EmbeddingManager
from app.rag.prompts.prompt_builder import PromptBuilder
from app.rag.rerankers.reranker_manager import RerankerManager
from app.rag.retrievers.retriever_manager import RetrieverManager


# Define the state schema
class RAGState(TypedDict):
    """State schema for the RAG pipeline."""

    question: str
    embedding: Optional[List[float]]
    retrieved_docs: List[tuple]
    reranked_docs: List[tuple]
    prompt: str
    messages: List[Dict[str, str]]
    response: str
    metadata: Dict


class RAGPipeline:
    """Complete RAG pipeline using LangGraph."""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            collection_name: Name of the vector store collection
            llm_model: LLM model name
            temperature: LLM temperature
            max_tokens: Maximum tokens for generation
            api_key: OpenAI API key (defaults to environment)
        """
        rag_config = get_rag_config()

        # Get API key
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable"
            )

        # Initialize components
        self.embedding_manager = EmbeddingManager(api_key=api_key)
        self.retriever = RetrieverManager(
            collection_name=collection_name, embedding_manager=self.embedding_manager
        )
        self.reranker = RerankerManager()
        self.prompt_builder = PromptBuilder()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model or rag_config.llm_model,
            temperature=temperature or rag_config.temperature,
            max_tokens=max_tokens or rag_config.max_tokens,
            openai_api_key=api_key,
        )

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create workflow
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("embed_question", self._embed_question)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("rerank", self._rerank_documents)
        workflow.add_node("create_prompt", self._create_prompt)
        workflow.add_node("generate_response", self._generate_response)

        # Define edges (pipeline flow)
        workflow.set_entry_point("embed_question")
        workflow.add_edge("embed_question", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "create_prompt")
        workflow.add_edge("create_prompt", "generate_response")
        workflow.add_edge("generate_response", END)

        # Compile the graph
        return workflow.compile()

    def _embed_question(self, state: RAGState) -> RAGState:
        """Embed the user's question."""
        question = state["question"]
        embedding = self.embedding_manager.embed_query(question)

        return {
            **state,
            "embedding": embedding,
            "metadata": {**state.get("metadata", {}), "embedding_dim": len(embedding)},
        }

    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents."""
        question = state["question"]
        retrieved_docs = self.retriever.retrieve(question)

        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "metadata": {
                **state.get("metadata", {}),
                "num_retrieved": len(retrieved_docs),
            },
        }

    def _rerank_documents(self, state: RAGState) -> RAGState:
        """Rerank retrieved documents."""
        question = state["question"]
        retrieved_docs = state["retrieved_docs"]

        reranked_docs = self.reranker.rerank(question, retrieved_docs)

        return {
            **state,
            "reranked_docs": reranked_docs,
            "metadata": {
                **state.get("metadata", {}),
                "num_reranked": len(reranked_docs),
            },
        }

    def _create_prompt(self, state: RAGState) -> RAGState:
        """Create the prompt from question and context."""
        question = state["question"]
        reranked_docs = state["reranked_docs"]

        # Convert reranked docs to expected format (remove rerank score)
        context_docs = [(doc[0], doc[1], doc[2]) for doc in reranked_docs]

        # Build messages for chat model
        messages = self.prompt_builder.build_rag_messages(
            question=question,
            context_documents=context_docs,
            include_metadata=True,
            include_scores=False,
        )

        # Also build string prompt for logging
        prompt = self.prompt_builder.build_rag_prompt(
            question=question, context_documents=context_docs
        )

        return {**state, "messages": messages, "prompt": prompt}

    def _generate_response(self, state: RAGState) -> RAGState:
        """Generate response using LLM."""
        messages = state["messages"]

        # Convert to LangChain message format
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                from langchain_core.messages import SystemMessage

                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                from langchain_core.messages import HumanMessage

                lc_messages.append(HumanMessage(content=msg["content"]))

        # Generate response
        response = self.llm.invoke(lc_messages)

        return {
            **state,
            "response": response.content,
            "metadata": {
                **state.get("metadata", {}),
                "response_length": len(response.content),
            },
        }

    def query(self, question: str) -> Dict:
        """Run a query through the RAG pipeline.

        Args:
            question: User's question

        Returns:
            Dictionary with response and metadata
        """
        # Initialize state
        initial_state = {
            "question": question,
            "embedding": None,
            "retrieved_docs": [],
            "reranked_docs": [],
            "prompt": "",
            "messages": [],
            "response": "",
            "metadata": {},
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        # Return response and metadata
        return {
            "question": final_state["question"],
            "response": final_state["response"],
            "context_documents": final_state["reranked_docs"],
            "metadata": final_state["metadata"],
        }

    def add_documents(
        self, documents: List[str], metadatas: Optional[List[Dict]] = None
    ):
        """Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        return self.retriever.add_documents(documents, metadatas)

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        collection_stats = self.retriever.get_collection_stats()
        return {
            "collection": collection_stats,
            "components": {
                "embedding_model": self.embedding_manager.model,
                "llm_model": self.llm.model_name,
                "rerank_model": self.reranker.model_name,
            },
        }


if __name__ == "__main__":
    # Example usage
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    # Add sample documents
    print("\nAdding documents to vector store...")
    documents = [
        "LangChain is a framework for developing applications powered by language models. It provides abstractions for working with LLMs, chains, and agents.",
        "RAG (Retrieval Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents and uses them as context for generating responses.",
        "Vector databases like ChromaDB and Pinecone store embeddings and enable efficient similarity search for semantic retrieval tasks.",
        "LangGraph extends LangChain with graph-based workflows. It allows building complex multi-step applications with state management and conditional logic.",
        "Reranking improves retrieval quality by using cross-encoder models to re-score documents based on their relevance to the query.",
        "Embeddings are dense vector representations of text that capture semantic meaning. They enable similarity-based search and retrieval.",
    ]

    metadatas = [
        {"source": "langchain_docs", "topic": "framework"},
        {"source": "rag_guide", "topic": "rag"},
        {"source": "vector_db_intro", "topic": "storage"},
        {"source": "langgraph_docs", "topic": "framework"},
        {"source": "reranking_paper", "topic": "retrieval"},
        {"source": "embedding_guide", "topic": "embedding"},
    ]

    pipeline.add_documents(documents, metadatas)

    # Run queries
    print("\n" + "=" * 80)
    print("Testing RAG Pipeline")
    print("=" * 80)

    queries = [
        "What is RAG and how does it work?",
        "How does LangGraph extend LangChain?",
        "Why is reranking important?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)

        result = pipeline.query(query)

        print(f"\nResponse:\n{result['response']}")
        print(f"\nMetadata: {result['metadata']}")
        print("\nTop context documents:")
        for i, doc in enumerate(result["context_documents"][:2], 1):
            print(f"{i}. [{doc[3]:.3f}] {doc[0][:100]}...")

    # Pipeline stats
    print("\n" + "=" * 80)
    stats = pipeline.get_stats()
    print("Pipeline Statistics:")
    print(f"  Collection: {stats['collection']}")
    print(f"  Components: {stats['components']}")
