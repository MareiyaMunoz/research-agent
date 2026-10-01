from app.agent.loop import run_agent

if __name__ == "__main__":
    question = (
        "What are the trade-offs between using pgvector inside PostgreSQL and a "
        "dedicated vector database like Pinecone for a small RAG app?"
    )
    print("\nFINAL ANSWER:\n", run_agent(question, max_steps=10))