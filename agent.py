# agent.py
import os
from flask import Flask, render_template, request
from smolagents import ToolCallingAgent, ToolCollection, LiteLLMModel
from mcp import StdioServerParameters

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

# -------------------------
# LLM Configuration
# -------------------------
model = LiteLLMModel(
    model_id=os.getenv("LITELLM_MODEL_ID", "ollama_chat/qwen2.5:14b"),
    num_ctx=int(os.getenv("LITELLM_CTX", "8192")),
)

server_parameters = StdioServerParameters(
    command=os.getenv("PY_CMD", "python"),
    args=[os.getenv("SERVER_FILE", "server.py")],
)

# -------------------------
# Home Page
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# -------------------------
# Form Submission
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    # Read user inputs
    gender = form["gender"]
    age = form["age"]
    height = form["height"]
    weight = form["weight"]
    duration = form["duration"]
    heart_rate = form["heart_rate"]
    body_temp = form["body_temp"]
    goal = form["goal"]

    # Prompt template
    prompt = f"""
You are a professional fitness assistant.

User details:
- Gender: {gender}
- Age: {age}
- Height: {height} cm
- Weight: {weight} kg
- Workout Duration: {duration} minutes
- Heart Rate: {heart_rate}
- Body Temperature: {body_temp}

Goal:
{goal}

Tasks:
1. Predict calories burnt.
2. Recommend 3 food sets to match calories (weight_difference_percentage=0).
3. Explain diet plan, macro targets, and daily guidance clearly.
"""

    # Run agent with MCP tools
    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        agent = ToolCallingAgent(
            tools=tool_collection.tools,
            model=model
        )
        result = agent.run(prompt)

    return render_template("result.html", result=result)

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
