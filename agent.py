# agent.py
import os
from smolagents import ToolCallingAgent, ToolCollection, LiteLLMModel
from mcp import StdioServerParameters

# Note:
# - Ensure server.py is in the same folder or provide full path in args.
# - If you prefer to run server separately, replace StdioServerParameters with connection params
#   that point to a running FastMCP instance (see smolagents docs).

# Configure LiteLLM model (adjust model_id to the one you have available)
model = LiteLLMModel(
    model_id=os.getenv("LITELLM_MODEL_ID", "ollama_chat/qwen2.5:14b"),
    num_ctx=int(os.getenv("LITELLM_CTX", "8192")),
)

# Start server.py as a subprocess and connect through FastMCP protocol
# command "python" is used. On some systems use "python3".
server_parameters = StdioServerParameters(
    command=os.getenv("PY_CMD", "python"),
    args=[os.getenv("SERVER_FILE", "server.py")],
)

# Create tool collection by launching mcp server (via Stdio)
with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = ToolCallingAgent(tools=tool_collection.tools, model=model)

    # Example prompt. Replace with interactive loop or orchestration you need.
    # The agent can call 'predict_calories' and 'recommend_foods' tools exposed by server.py.
    prompt = (
        "You are a fitness assistant. User data: Gender=1, Age=28, Height=175, "
        "Weight=72, Duration=45, Heart_Rate=140, Body_Temp=36.6. "
        "Predict calories burnt, then recommend 3 food items (3 sets) to match the calories "
        "with weight_difference_percentage=0. Explain me the diet plan and targets in detail"
    )

    # Run the agent. This will let the agent call tools and return a final response.
    agent.run(prompt)
