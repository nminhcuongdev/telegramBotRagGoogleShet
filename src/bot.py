import asyncio
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

# ChromaDB config
CHROMA_DIR = "data/chroma_db_checkmate"
COLLECTION_NAME = "checkmate_customers"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize RAG chain globally
rag_chain = None
retriever = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = """
**Welcome to Checkmate Query Bot!**

I can help you look up customer information from the Checkmate report.

**Examples of questions I can answer:**
- Customer balance: "What is the balance of C550?"
- Assigned tech: "Which customers are managed by Tech Hai?"
- Spending: "How much did C548 P3 spend?"
- Statistics: "Which customer has the highest balance?"
- Summary: "What is the total balance of customers under Tech An?"

**How to use:**
Just send your question and I will search and respond.

Ask me anything to begin.
    """
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_name = update.message.from_user.first_name or "User"
    print(f"[LOG] {user_name} asked: {user_text}")

    try:
        # Show typing indicator
        await update.message.chat.send_action("typing")

        # Get relevant documents for transparency
        docs = retriever.invoke(user_text)
        print(f"[LOG] Found {len(docs)} relevant documents")

        # Query RAG chain
        answer = rag_chain.invoke(user_text)

        # Format response
        response = f"**Answer:**\n\n{answer}"

        # Send response (split if too long)
        if len(response) > 4096:
            await update.message.reply_text(
                f"**Answer:**\n\n{answer}", parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(response, parse_mode="Markdown")

    except Exception as e:
        print(f"[ERROR] {e}")
        error_msg = (
            f"**Error while processing your question**\n\n`{str(e)}`\n\n"
            "Try these:\n"
            "- Ask a clearer question\n"
            "- Use an exact customer code (for example: C550)\n"
            "- Ask about information that exists in the database"
        )
        await update.message.reply_text(error_msg, parse_mode="Markdown")


def create_rag_chain():
    """Create a RAG chain for querying Checkmate data."""
    print("Initializing RAG chain...")

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
Use only the provided context to answer the question accurately.

Column definitions:
- Name: Customer name or code
- Tech: Assigned technical owner
- Account: Assigned accountant
- Balance: Current customer balance
- Total Customer Spend: Total customer spending for the month

Database context:
{context}

Question: {question}

If the context is not sufficient, say you do not know.
Answer in clear and concise English.

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def initialize_rag():
    """Initialize RAG chain on startup."""
    global rag_chain, retriever
    try:
        if not os.path.exists(CHROMA_DIR):
            print(f"Warning: ChromaDB not found at {CHROMA_DIR}")
            print("   Run 'python test_ingest_csv.py' first to create the database.")
            return False

        rag_chain, retriever = create_rag_chain()
        print("RAG chain initialized successfully with ChromaDB")
        return True
    except Exception as e:
        print(f"Failed to initialize RAG: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("CHECKMATE TELEGRAM BOT")
    print("=" * 60)

    if not initialize_rag():
        print("\nCannot start bot without RAG initialization.")
        print("\nTo fix this:")
        print("   1. Run: python test_ingest_csv.py")
        print("   2. Wait for ChromaDB to be created")
        print("   3. Run this bot again")
        return

    print("\nConnecting to Telegram...")
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running and ready to receive questions.")
    print("Open Telegram and find your bot to start.")
    print("\nPress Ctrl+C to stop\n")
    print("=" * 60)

    app.run_polling()


if __name__ == "__main__":
    main()
