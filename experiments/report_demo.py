"""Full Phase 5 demo: run the agent, then render a cited Markdown report."""

from app.agent.loop import run_agent
from app.report import render_report

question = (
    "What are the pros and cons of using Redis versus PostgreSQL for "
    "session storage in a web application?"
)

result = run_agent(question, max_steps=10)

print("\n" + "=" * 70)
print(render_report(question, result))
print("=" * 70)
print(f"\nsteps used: {result.steps}, reached limit: {result.reached_limit}")
print("trace:")
for rec in result.trace:
    print(f"  {rec}")
