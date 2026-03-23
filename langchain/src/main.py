from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain.agents.structured_output import ToolStrategy


SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""


# Define tools
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool
def get_user_location() -> str:
    """Retrieve user information based on user ID."""
    return "Florida"


# Setup model (Ollama)
model = ChatOllama(
    model="gpt-oss:20b-cloud", temperature=0
)  # IMPORTANT: supports tools


# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""

    punny_response: str
    weather_conditions: str | None = None


# Add memory
memory = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    tools=[get_weather, get_user_location],
    system_prompt=SYSTEM_PROMPT,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=memory,
)

# Run agent (conversation)
config = {"configurable": {"thread_id": "1"}}

# First message
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Outside?"}]},
    config=config,
)
print(response1)

# Second message (memory test)
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "thanks!"}]}, config=config
)
print(response2)
