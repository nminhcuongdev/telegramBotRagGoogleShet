"""
Query Checkmate data using RAG (Retrieval-Augmented Generation).
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# =========================
# CONFIG
# =========================
CHROMA_DIR = "data/chroma_db_checkmate"
COLLECTION_NAME = "checkmate_customers"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# =========================
# RAG SYSTEM
# =========================
def create_rag_chain():
    """Create a RAG chain for querying Checkmate data."""

    # Load ChromaDB vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 500})

    # Prompt template
    template = """You are a Checkmate data analysis assistant.
Use only the provided context to answer accurately and clearly.

Column definitions:
- Name: Customer name or code
- Tech: Assigned technical owner
- Account: Assigned accountant
- Balance: Customer account balance
- Total Customer Spend: Total amount spent by the customer this month

Context from database:
{context}

Question: {question}

If the information is insufficient, explicitly say you do not know.
Answer in concise English.

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Create LLM
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    # Create RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def query_checkmate_data(question: str):
    """Query the Checkmate data."""
    print(f"\nQuestion: {question}")
    print("=" * 60)

    # Create RAG chain
    rag_chain, retriever = create_rag_chain()

    # Retrieve relevant documents
    docs = retriever.invoke(question)
    print(f"\nFound {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(doc.page_content[:150] + "...")
        print(f"Metadata: {doc.metadata}")

    # Get answer from RAG
    print("\n" + "=" * 60)
    print("Answer:")
    answer = rag_chain.invoke(question)
    print(answer)
    print("=" * 60)

    return answer


# =========================
# EXAMPLE QUERIES
# =========================
def run_examples():
    """Run example queries."""
    example_questions = [
        "What is the balance of customer C550?",
        "Which customers are managed by Tech Hai?",
        "Which customer has the highest balance?",
        "How much did customer C548 P3 spend in February?",
        "How many customers does Tech Hoang manage?",
    ]

    for question in example_questions:
        query_checkmate_data(question)
        print("\n" + "=" * 60 + "\n")


# =========================
# INTERACTIVE MODE
# =========================
def interactive_mode():
    """Run in interactive mode."""
    print("\n" + "=" * 60)
    print("CHECKMATE DATA QUERY SYSTEM")
    print("Type your question (or 'exit' to quit)")
    print("=" * 60)

    rag_chain, retriever = create_rag_chain()

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            # Retrieve documents
            docs = retriever.invoke(question)
            print(f"\nFound {len(docs)} relevant documents")

            # Get answer
            answer = rag_chain.invoke(question)
            print(f"\nAnswer:\n{answer}")

        except Exception as e:
            print(f"Error: {e}")


# =========================
# MAIN
# =========================
def main():
    import sys

    if len(sys.argv) > 1:
        # Query from command line
        question = " ".join(sys.argv[1:])
        query_checkmate_data(question)
    else:
        # Interactive mode
        print("\nChoose mode:")
        print("1. Run sample questions")
        print("2. Interactive mode (type your own questions)")
        choice = input("\nSelect (1/2): ").strip()

        if choice == "1":
            run_examples()
        else:
            interactive_mode()


if __name__ == "__main__":
    main()
