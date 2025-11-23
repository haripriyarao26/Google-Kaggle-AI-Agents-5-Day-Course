"""
Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

🚀 Agent Tool Patterns and Best Practices

Welcome to Day-2 of the Kaggle 5-day Agents course!

In the previous notebook, you learned how to add custom Python functions as tools to your agent. 
In this script, we'll take the next step: consuming external MCP services and handling 
long-running operations.

In this script, you'll learn how to:
- ✅ Connect to external MCP servers
- ✅ Implement long-running operations that can pause agent execution for external input
- ✅ Build resumable workflows that maintain state across conversation breaks
- ✅ Understand when and how to use these patterns
"""

import os
import uuid
import asyncio
import base64

# ============================================================================
# Section 1: Setup
# ============================================================================

# 1.1: Install dependencies
# Run: pip install google-adk

# 1.2: Configure your Gemini API Key
# 
# This script uses the Gemini API, which requires an API key.
# 
# 1. Get your API key: Create an API key in Google AI Studio
#    https://aistudio.google.com/app/api-keys
# 
# 2. Set the API key as an environment variable:
#    export GOOGLE_API_KEY="your-api-key-here"
# 
# For Kaggle Notebooks, use this instead:
# from kaggle_secrets import UserSecretsClient
# try:
#     GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
#     os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
#     print("✅ Setup and authentication complete.")
# except Exception as e:
#     print(f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")

# For local use, set the API key from environment variable
if "GOOGLE_API_KEY" not in os.environ:
    print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables.")
    print("   Please set it using: export GOOGLE_API_KEY='your-api-key'")
else:
    print("✅ Setup and authentication complete.")


# 1.3: Import ADK components
from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

print("✅ ADK components imported successfully.")


# 1.4: Configure Retry Options
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# ============================================================================
# Section 2: Model Context Protocol
# ============================================================================

# Model Context Protocol (MCP) is an open standard that lets agents use 
# community-built integrations. Instead of writing your own integrations and 
# API clients, just connect to an existing MCP server.
# 
# MCP enables agents to:
# ✅ Access live, external data from databases, APIs, and services without custom integration code
# ✅ Leverage community-built tools with standardized interfaces
# ✅ Scale capabilities by connecting to multiple specialized servers

# 2.1: How MCP Works
# 
# MCP connects your agent (the client) to external MCP servers that provide tools:
# - MCP Server: Provides specific tools (like image generation, database access)
# - MCP Client: Your agent that uses those tools
# - All servers work the same way - standardized interface

# 2.2: Using MCP with Your Agent
# 
# The workflow is simple:
# 1. Choose an MCP Server and tool
# 2. Create the MCP Toolset (configure connection)
# 3. Add it to your agent
# 4. Run and test the agent

# Step 1: Choose MCP Server
# For this demo, we'll use the Everything MCP Server - an npm package designed 
# for testing MCP integrations. It provides a getTinyImage tool that returns 
# a simple test image (16x16 pixels, Base64-encoded).
# 
# Find more servers: https://modelcontextprotocol.io/examples
# 
# NOTE: This is a demo server to learn MCP. In production, you'll use servers 
# for Google Maps, Slack, Discord, etc.

# Step 2: Create the MCP Toolset
# The McpToolset is used to integrate an ADK Agent with an MCP Server.
# 
# What the code does:
# - Uses npx (Node package runner) to run the MCP server
# - Connects to @modelcontextprotocol/server-everything
# - Filters to only use the getTinyImage tool

mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",  # Run MCP server via npx
            args=[
                "-y",  # Argument for npx to auto-confirm install
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"],
        ),
        timeout=30,
    )
)

print("✅ MCP Tool created")

# Behind the scenes:
# 1. Server Launch: ADK runs npx -y @modelcontextprotocol/server-everything
# 2. Handshake: Establishes stdio communication channel
# 3. Tool Discovery: Server tells ADK: "I provide getTinyImage" functionality
# 4. Integration: Tools appear in agent's tool list automatically
# 5. Execution: When agent calls getTinyImage(), ADK forwards to MCP server
# 6. Response: Server result is returned to agent seamlessly
# 
# Why This Matters: You get instant access to tools without writing integration code!

# Step 3: Add MCP tool to agent
# Create image agent with MCP integration
image_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="image_agent",
    instruction="Use the MCP Tool to generate images for user queries",
    tools=[mcp_image_server],
)

# Create the runner
runner = InMemoryRunner(agent=image_agent)


async def test_mcp_agent():
    """Test the MCP agent with image generation."""
    print("\n" + "="*70)
    print("Testing MCP Agent - Image Generation")
    print("="*70 + "\n")
    
    # Step 4: Test the agent
    response = await runner.run_debug("Provide a sample tiny image", verbose=True)
    
    # Display the image (if available)
    # Note: In a script environment, you may need to save the image to a file
    # instead of displaying it
    for event in response:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    for item in part.function_response.response.get("content", []):
                        if item.get("type") == "image":
                            # In a script, you might save this to a file
                            image_data = base64.b64decode(item["data"])
                            print(f"✅ Image received (size: {len(image_data)} bytes)")
                            # Uncomment to save the image:
                            # with open("generated_image.png", "wb") as f:
                            #     f.write(image_data)
                            # print("Image saved to generated_image.png")


# 2.3: Extending to Other MCP Servers
# 
# The same pattern works for any MCP server - only the connection_params change.
# 
# Example: Kaggle MCP Server
# McpToolset(
#     connection_params=StdioConnectionParams(
#         server_params=StdioServerParameters(
#             command='npx',
#             args=['-y', 'mcp-remote', 'https://www.kaggle.com/mcp'],
#         ),
#         timeout=30,
#     )
# )
# 
# Example: GitHub MCP Server
# McpToolset(
#     connection_params=StreamableHTTPServerParams(
#         url="https://api.githubcopilot.com/mcp/",
#         headers={
#             "Authorization": f"Bearer {GITHUB_TOKEN}",
#             "X-MCP-Toolsets": "all",
#             "X-MCP-Readonly": "true"
#         },
#     ),
# )


# ============================================================================
# Section 3: Long-Running Operations (Human-in-the-Loop)
# ============================================================================

# So far, all tools execute and return immediately:
# User asks → Agent calls tool → Tool returns result → Agent responds
# 
# But what if your tools are long-running or you need human approval before 
# completing an action?
# 
# Example: A shipping agent should ask for approval before placing a large order.
# 
# User asks → Agent calls tool → Tool PAUSES and asks human → Human approves → 
# Tool completes → Agent responds
# 
# This is called a Long-Running Operation (LRO) - the tool needs to pause, wait 
# for external input (human approval), then resume.

# When to use Long-Running Operations:
# - 💰 Financial transactions requiring approval (transfers, purchases)
# - 🗑️ Bulk operations (delete 1000 records - confirm first!)
# - 📋 Compliance checkpoints (regulatory approval needed)
# - 💸 High-cost actions (spin up 50 servers - are you sure?)
# - ⚠️ Irreversible operations (permanently delete account)

# 3.1: What We're Building Today
# 
# Let's build a shipping coordinator agent with one tool that:
# - Auto-approves small orders (≤5 containers)
# - Pauses and asks for approval on large orders (>5 containers)
# - Completes or cancels based on the approval decision
# 
# This demonstrates the core long-running operation pattern: 
# pause → wait for human input → resume

# 3.2: The Shipping Tool with Approval Logic
# 
# The ToolContext Parameter:
# Notice the function signature includes tool_context: ToolContext. ADK automatically 
# provides this object when your tool runs. It gives you two key capabilities:
# 1. Request approval: Call tool_context.request_confirmation()
# 2. Check approval status: Read tool_context.tool_confirmation

LARGE_ORDER_THRESHOLD = 5


def place_shipping_order(
    num_containers: int, destination: str, tool_context: ToolContext
) -> dict:
    """Places a shipping order. Requires approval if ordering more than 5 containers.

    Args:
        num_containers: Number of containers to ship
        destination: Shipping destination
        tool_context: ADK-provided context for tool execution

    Returns:
        Dictionary with order status
    """
    # SCENARIO 1: Small orders (≤5 containers) auto-approve
    if num_containers <= LARGE_ORDER_THRESHOLD:
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-AUTO",
            "num_containers": num_containers,
            "destination": destination,
            "message": f"Order auto-approved: {num_containers} containers to {destination}",
        }

    # SCENARIO 2: This is the first time this tool is called. Large orders need 
    # human approval - PAUSE here.
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Large order: {num_containers} containers to {destination}. Do you want to approve?",
            payload={"num_containers": num_containers, "destination": destination},
        )
        return {  # This is sent to the Agent
            "status": "pending",
            "message": f"Order for {num_containers} containers requires approval",
        }

    # SCENARIO 3: The tool is called AGAIN and is now resuming. Handle approval 
    # response - RESUME here.
    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-HUMAN",
            "num_containers": num_containers,
            "destination": destination,
            "message": f"Order approved: {num_containers} containers to {destination}",
        }
    else:
        return {
            "status": "rejected",
            "message": f"Order rejected: {num_containers} containers to {destination}",
        }


print("✅ Long-running functions created!")

# 3.3: Understanding the Code
# 
# How the Three Scenarios Work:
# 
# Scenario 1: Small order (≤5 containers): Returns immediately with auto-approved status.
# 
# Scenario 2: Large order - FIRST CALL
# - Tool detects it's a first call: if not tool_context.tool_confirmation:
# - Calls request_confirmation() to request human approval
# - Returns {'status': 'pending', ...} immediately
# - ADK automatically creates adk_request_confirmation event
# - Agent execution pauses - waiting for human decision
# 
# Scenario 3: Large order - RESUMED CALL
# - Tool detects it's resuming: if not tool_context.tool_confirmation: is now False
# - Checks human decision: tool_context.tool_confirmation.confirmed
# - If True → Returns approved status
# - If False → Returns rejected status
# 
# Key insight: Between the two calls, your workflow code (in Section 4) must detect 
# the adk_request_confirmation event and resume with the approval decision.

# 3.4: Create the Agent, App and Runner

# Step 1: Create the agent
# Add the tool to the Agent. The tool decides internally when to request approval 
# based on the order size.
shipping_agent = LlmAgent(
    name="shipping_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a shipping coordinator assistant.
  
  When users request to ship containers:
   1. Use the place_shipping_order tool with the number of containers and destination
   2. If the order status is 'pending', inform the user that approval is required
   3. After receiving the final result, provide a clear summary including:
      - Order status (approved/rejected)
      - Order ID (if available)
      - Number of containers and destination
   4. Keep responses concise but informative
  """,
    tools=[FunctionTool(func=place_shipping_order)],
)

print("✅ Shipping Agent created!")

# Step 2: Wrap in resumable App
# 
# The problem: A regular LlmAgent is stateless - each call is independent with 
# no memory of previous interactions. If a tool requests approval, the agent can't 
# remember what it was doing.
# 
# The solution: Wrap your agent in an App with resumability enabled. The App adds 
# a persistence layer that saves and restores state.
# 
# What gets saved when a tool pauses:
# - All conversation messages so far
# - Which tool was called (place_shipping_order)
# - Tool parameters (10 containers, Rotterdam)
# - Where exactly it paused (waiting for approval)
# 
# When you resume, the App loads this saved state so the agent continues exactly 
# where it left off - as if no time passed.

shipping_app = App(
    name="shipping_coordinator",
    root_agent=shipping_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

print("✅ Resumable app created!")

# Step 3: Create Session and Runner with the App
# Pass app=shipping_app instead of agent=... so the runner knows about resumability.
session_service = InMemorySessionService()

# Create runner with the resumable app
shipping_runner = Runner(
    app=shipping_app,  # Pass the app instead of the agent
    session_service=session_service,
)

print("✅ Runner created!")


# ============================================================================
# Section 4: Building the Workflow
# ============================================================================

# The Critical Part - Handling Events in Your Workflow
# 
# The agent won't automatically handle pause/resume. Every long-running operation 
# workflow requires you to:
# 1. Detect the pause: Check if events contain adk_request_confirmation
# 2. Get human decision: In production, show UI and wait for user click. Here, we simulate it.
# 3. Resume the agent: Send the decision back with the saved invocation_id

# Understand Key Technical Concepts:
# 
# events - ADK creates events as the agent executes. Tool calls, model responses, 
# function results - all become events
# 
# adk_request_confirmation event - This event is special - it signals "pause here!"
# - Automatically created by ADK when your tool calls request_confirmation()
# - Contains the invocation_id
# - Your workflow must detect this event to know the agent paused
# 
# invocation_id - Every call to run_async() gets a unique invocation_id (like "abc123")
# - When a tool pauses, you save this ID
# - When resuming, pass the same ID so ADK knows which execution to continue
# - Without it, ADK would start a NEW execution instead of resuming the paused one

# 4.3: Helper Functions to Process Events

def check_for_approval(events):
    """Check if events contain an approval request.

    Returns:
        dict with approval details or None
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None


def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")


def create_approval_response(approval_info, approved):
    """Create approval response message."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )


print("✅ Helper functions defined")

# 4.4: The Workflow Function
async def run_shipping_workflow(query: str, auto_approve: bool = True):
    """Runs a shipping workflow with approval handling.

    Args:
        query: User's shipping request
        auto_approve: Whether to auto-approve large orders (simulates human decision)
    """
    print(f"\n{'='*60}")
    print(f"User > {query}\n")

    # Generate unique session ID
    session_id = f"order_{uuid.uuid4().hex[:8]}"

    # Create session
    await session_service.create_session(
        app_name="shipping_coordinator", user_id="test_user", session_id=session_id
    )

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    # STEP 1: Send initial request to the Agent. If num_containers > 5, the Agent 
    # returns the special adk_request_confirmation event
    async for event in shipping_runner.run_async(
        user_id="test_user", session_id=session_id, new_message=query_content
    ):
        events.append(event)

    # STEP 2: Loop through all the events generated and check if 
    # adk_request_confirmation is present.
    approval_info = check_for_approval(events)

    # STEP 3: If the event is present, it's a large order - HANDLE APPROVAL WORKFLOW
    if approval_info:
        print(f"⏸️  Pausing for approval...")
        print(f"🤔 Human Decision: {'APPROVE ✅' if auto_approve else 'REJECT ❌'}\n")

        # PATH A: Resume the agent by calling run_async() again with the approval decision
        async for event in shipping_runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=create_approval_response(approval_info, auto_approve),
            invocation_id=approval_info["invocation_id"],  # Critical: same invocation_id tells ADK to RESUME
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent > {part.text}")

    else:
        # PATH B: If the adk_request_confirmation is not present - no approval needed 
        # - order completed immediately.
        print_agent_response(events)

    print(f"{'='*60}\n")


print("✅ Workflow function ready")


async def main():
    """Main function to run all examples."""
    
    # Test MCP agent (optional - requires npx and npm)
    # await test_mcp_agent()
    
    # Demo 1: It's a small order. Agent receives auto-approved status from tool
    await run_shipping_workflow("Ship 3 containers to Singapore")

    # Demo 2: Workflow simulates human decision: APPROVE ✅
    await run_shipping_workflow("Ship 10 containers to Rotterdam", auto_approve=True)

    # Demo 3: Workflow simulates human decision: REJECT ❌
    await run_shipping_workflow("Ship 8 containers to Los Angeles", auto_approve=False)
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

