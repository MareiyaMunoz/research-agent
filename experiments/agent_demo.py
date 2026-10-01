from app.agent.loop import run_agent

if __name__ == "__main__":
    question = "What's the weather in Manila and Tokyo? Give me both in Fahrenheit."
    print("\nFINAL ANSWER:\n", run_agent(question))