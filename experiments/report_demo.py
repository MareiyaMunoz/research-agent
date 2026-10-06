"""Full Phase 5 demo: run the agent, then render a cited Markdown report."""

from app.agent.loop import run_agent
from app.report import render_report

question = (
    "What are the trade-offs between using pgvector inside PostgreSQL and a "
    "dedicated vector database like Pinecone for a small RAG app?"
)

result = run_agent(question, max_steps=10)

print("\n" + "=" * 70)
print(render_report(question, result))
print("=" * 70)
print(f"\nsteps used: {result.steps}, reached limit: {result.reached_limit}")
