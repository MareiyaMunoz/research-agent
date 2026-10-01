from google.genai import types

from app.agent.tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS
from app.llm import MODEL, generate

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when you need "
    "information. Do not guess values you can look up. When you have enough "
    "information, give a short, clear final answer."
    "If several lookups are independent of each other, request them all in the same step."
)

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def run_tool(name: str, args: dict) -> dict:
    """Run one tool safely. Errors become data the model can read."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return func(**args)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def run_agent(question: str, max_steps: int = 8) -> str:
    # The conversation history, which grows every round
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    for step in range(1, max_steps + 1):
        response = generate(model=MODEL, contents=contents, config=config)

        # Save the model's turn exactly as returned
        contents.append(response.candidates[0].content)

        # No tool requests -> this is the final answer
        if not response.function_calls:
            return response.text

        # Otherwise run every requested tool and collect the results
        result_parts = []
        for call in response.function_calls:
            print(f"[step {step}] calling {call.name}({dict(call.args)})")
            result = run_tool(call.name, dict(call.args))
            print(f"[step {step}] result: {result}")
            result_parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": result},
                )
            )

        # Send all results back in one message, then loop again
        contents.append(types.Content(role="user", parts=result_parts))

    return "Stopped: reached the maximum number of steps without a final answer."