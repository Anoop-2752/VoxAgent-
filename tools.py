"""
VoxAgent - Tools
Real-world actions the agent can take. Both are free, no API key required.
"""

import requests
from langchain_core.tools import tool
from ddgs import DDGS


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a city or location name. Use this whenever
    the user asks about weather, temperature, or whether they need a jacket/umbrella."""
    print(f"[tool] get_weather('{location}')")
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
            timeout=5,
        ).json()

        if not geo.get("results"):
            return f"Could not find a location matching '{location}'."

        place = geo["results"][0]
        lat, lon = place["latitude"], place["longitude"]
        place_name = place.get("name", location)
        country = place.get("country", "")

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=5,
        ).json()

        current = weather.get("current_weather", {})
        temp = current.get("temperature")
        wind = current.get("windspeed")

        return f"Current weather in {place_name}, {country}: {temp}°C, wind speed {wind} km/h."
    except Exception as e:
        return f"Weather lookup failed: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for current information, facts, or anything time-sensitive
    that you can't answer reliably from memory alone."""
    print(f"[tool] web_search('{query}')")
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No search results found."

        return "\n".join(f"- {r['title']}: {r['body']}" for r in results)
    except Exception as e:
        return f"Web search failed: {e}"