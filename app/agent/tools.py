# --- The actual functions (your code runs these) ---

def get_weather(city: str) -> dict:
    fake_data = {"manila": 31, "tokyo": 22, "london": 15}  # fake data
    temp = fake_data.get(city.lower())
    if temp is None:
        return {"error": f"No weather data for {city}"}
    return {"city": city, "temp_c": temp}


def celsius_to_fahrenheit(celsius: float) -> dict:
    return {"fahrenheit": round(celsius * 9 / 5 + 32, 1)}


# --- Registry: tool name -> function ---
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "celsius_to_fahrenheit": celsius_to_fahrenheit,
}

# --- Descriptions the model reads ---
TOOL_DECLARATIONS = [
    {
        "name": "get_weather",
        "description": "Get the current temperature in Celsius for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Manila"}
            },
            "required": ["city"],
        },
    },
    {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a temperature from Celsius to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {"type": "number", "description": "Temperature in Celsius"}
            },
            "required": ["celsius"],
        },
    },
]