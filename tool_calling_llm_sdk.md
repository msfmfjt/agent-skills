# Implementing Tool Calling for Non-Native LLMs (LM Studio--style Approach)

This document explains how tool calling can be implemented **at the SDK
layer** even when a model does not natively support tool calling. The
explanation is based on publicly observable behavior of systems like LM
Studio.

The internal source code of LM Studio is not public, so this explanation
reconstructs the architecture from documentation and observed behavior.

------------------------------------------------------------------------

# Architecture Overview

At a high level, the system likely consists of four layers.

    Client
      ↓
    OpenAI-Compatible API
      ↓
    Prompt / Tool Adapter
      ↓
    Local Model Inference
      ↓
    Output Parser
      ↓
    Client executes tool
      ↓
    Model final answer

------------------------------------------------------------------------

# 1. OpenAI-Compatible Request Layer

The server accepts OpenAI-style requests such as:

-   `messages`
-   `tools`
-   `tool_choice`

Example request:

``` json
{
  "messages": [
    {"role": "user", "content": "What is the weather in Tokyo?"}
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get weather information",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"}
        },
        "required": ["city"]
      }
    }
  ]
}
```

The server converts this request into an **internal representation**
rather than passing it directly to the model.

------------------------------------------------------------------------

# 2. Prompt / Template Adaptation Layer

This layer adapts tool definitions for the target model.

## Native Tool Support

If the model already supports tool calling (for example Qwen or Llama
3.1 with tool format), the server can use a **model-specific chat
template**.

## Fallback Tool Support

If the model does **not** support tool calling natively, the server
emulates it.

The adapter typically:

-   injects a special system prompt
-   lists available tools
-   forces the model to output a strict tool-call format
-   converts unsupported roles (like `tool`) into something the model
    understands

Example system prompt:

    You are a helpful assistant.

    If a tool is required, output exactly one tool request using:

    [TOOL_REQUEST]{"name":"tool_name","arguments":{...}}[END_TOOL_REQUEST]

    Available tools:
    - get_weather(city)

------------------------------------------------------------------------

# 3. Output Parsing Layer

After the model generates text, the server determines whether the
response is:

1.  A normal assistant response
2.  A tool request

Example tool request format:

    [TOOL_REQUEST]{"name":"get_weather","arguments":{"city":"Tokyo"}}[END_TOOL_REQUEST]

The parser extracts the JSON payload and converts it into an
OpenAI-style response.

Example converted output:

``` json
{
  "tool_calls": [
    {
      "name": "get_weather",
      "arguments": {
        "city": "Tokyo"
      }
    }
  ]
}
```

If parsing fails, the output is treated as normal text.

------------------------------------------------------------------------

# 4. Client-Side Execution Loop

The server typically **does not execute the tool itself**.

Instead, the client performs the execution loop:

    Client sends request
    ↓
    Server returns tool_calls
    ↓
    Client executes tool
    ↓
    Client sends tool result
    ↓
    Server asks model for final answer

Example flow:

    User: What is the weather in Tokyo?

    Model → tool request

    Client executes get_weather()

    Client sends tool result

    Model → final natural language answer

------------------------------------------------------------------------

# Native vs Fallback Tool Calling

## Native Tool Calling

Advantages:

-   Structured outputs
-   More reliable
-   Model was trained for this format

Example models:

-   Qwen
-   Llama 3.1+
-   Mistral tool models

------------------------------------------------------------------------

## Fallback Tool Calling

The SDK simulates tool calling by:

-   injecting tool descriptions
-   enforcing a strict output wrapper
-   parsing the wrapper afterward

Advantages:

-   Works with almost any LLM

Disadvantages:

-   Less reliable than native tool calling

------------------------------------------------------------------------

# Minimal Python Implementation

The following Python example demonstrates a simplified implementation.

Features:

-   Tool registry
-   Prompt renderer
-   Tool call parser
-   Tool executor
-   Agent loop

------------------------------------------------------------------------

## Python Code

``` python
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

TOOL_REQUEST_PATTERN = re.compile(
    r"\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]",
    re.DOTALL,
)

@dataclass
class Tool:
    name: str
    description: str
    schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]

class ToolCallParser:
    @staticmethod
    def parse(text: str) -> Optional[Dict[str, Any]]:
        match = TOOL_REQUEST_PATTERN.search(text)
        if not match:
            return None

        payload = match.group(1).strip()
        data = json.loads(payload)

        return {
            "name": data["name"],
            "arguments": data.get("arguments", {})
        }

class ToolExecutor:
    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}

    def execute(self, tool_call: Dict[str, Any]):
        tool = self.tools[tool_call["name"]]
        return tool.handler(tool_call["arguments"])

class PromptRenderer:
    @staticmethod
    def render_system_prompt(tools: List[Tool]) -> str:
        tool_desc = []
        for t in tools:
            tool_desc.append(json.dumps({
                "name": t.name,
                "description": t.description,
                "parameters": t.schema
            }, indent=2))

        return f"""
You are a helpful assistant.

If a tool is required output:

[TOOL_REQUEST]{{"name":"tool_name","arguments":{{...}}}}[END_TOOL_REQUEST]

Available tools:
{chr(10).join(tool_desc)}
"""

class Agent:
    def __init__(self, tools: List[Tool]):
        self.tools = tools
        self.executor = ToolExecutor(tools)

    def run(self, user_message: str):
        system_prompt = PromptRenderer.render_system_prompt(self.tools)

        response = mock_llm(system_prompt, user_message)

        tool_call = ToolCallParser.parse(response)

        if tool_call is None:
            return response

        result = self.executor.execute(tool_call)

        final = mock_llm(
            "Use the tool result to answer the user.",
            f"Tool result: {json.dumps(result)}"
        )

        return final

# Example tool

def get_weather(args):
    city = args["city"]
    return {"city": city, "temperature": 22, "condition": "Sunny"}

tools = [
    Tool(
        name="get_weather",
        description="Get weather for a city",
        schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        },
        handler=get_weather
    )
]

# Mock LLM

def mock_llm(system_prompt, user_text):
    if "weather" in user_text.lower():
        return '[TOOL_REQUEST]{"name":"get_weather","arguments":{"city":"Tokyo"}}[END_TOOL_REQUEST]'
    if "Tool result" in user_text:
        data = json.loads(user_text.split(":",1)[1])
        return f"The weather in {data['city']} is {data['condition']}."
    return "I can answer directly."

if __name__ == "__main__":
    agent = Agent(tools)
    print(agent.run("What is the weather in Tokyo?"))
```

------------------------------------------------------------------------

# Example Output

    The weather in Tokyo is Sunny.

------------------------------------------------------------------------

# Key Idea

This implementation demonstrates how an SDK can add tool calling
**without native model support**.

Core idea:

    tools → prompt injection
    model output → strict parsing
    tool execution → external
    result → fed back to model

This pattern allows tool calling to work with **almost any LLM**.
