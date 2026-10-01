from google.genai import types

from app.llm import MODEL, generate


# 1. A fake tool: a normal Python function
def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 31, "condition": "sunny"}  # fake data


# 2. Describe the tool to the model (it never sees your Python code, only this)
weather_tool = types.Tool(function_declarations=[{
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. Manila"}
        },
        "required": ["city"],
    },
}])

# 3. Turn OFF automatic function calling so we can see and handle it ourselves
config = types.GenerateContentConfig(
    tools=[weather_tool],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

question = "What's the weather like in Manila right now?"

response = generate(model=MODEL, contents=question, config=config)

# 4. Did the model answer, or ask for a tool?
if response.function_calls:
    call = response.function_calls[0]
    print("Model wants to call:", call.name)
    print("With arguments:", call.args)

    # 5. YOUR code runs the function
    result = get_weather(**call.args)
    print("Tool result:", result)

    # 6. Send the whole conversation back, including the tool result
    contents = [
        types.Content(role="user", parts=[types.Part(text=question)]),
        response.candidates[0].content,  # the model's tool request
        types.Content(
            role="user",
            parts=[types.Part.from_function_response(
                name=call.name,
                response={"result": result},
            )],
        ),
    ]

    final = generate(model=MODEL, contents=contents, config=config)
    print("Final answer:", final.text)

else:
    print("Model answered directly:", response.text)