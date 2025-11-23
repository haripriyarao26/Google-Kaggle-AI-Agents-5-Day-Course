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

🚀 Your First AI Agent: From Prompt to Action

Welcome to the Kaggle 5-day Agents course!

This script is your first step into building AI agents. An agent can do more than just 
respond to a prompt — it can take actions to find information or get things done.

In this script, you'll:
- ✅ Install Agent Development Kit (ADK) - run: pip install google-adk
- ✅ Configure your API key to use the Gemini model
- ✅ Build your first simple agent
- ✅ Run your agent and watch it use a tool (like Google Search) to answer a question
"""

import os
import asyncio

# ============================================================================
# Section 1: Setup
# ============================================================================

# 1.1 Configure your Gemini API Key
# 
# This script uses the Gemini API, which requires authentication.
# 
# 1. Get your API key: Create an API key in Google AI Studio
#    https://aistudio.google.com/app/api-keys
# 
# 2. Set the API key as an environment variable:
#    export GOOGLE_API_KEY="your-api-key-here"
# 
#    Or set it directly in this script (not recommended for production):
#    os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

# For Kaggle Notebooks, use this instead:
# from kaggle_secrets import UserSecretsClient
# try:
#     GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
#     os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
#     os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
#     print("✅ Gemini API key setup complete.")
# except Exception as e:
#     print(f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")

# For local use, set the API key from environment variable
if "GOOGLE_API_KEY" not in os.environ:
    print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables.")
    print("   Please set it using: export GOOGLE_API_KEY='your-api-key'")
else:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("✅ Gemini API key setup complete.")


# 1.2 Import ADK components
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

print("✅ ADK components imported successfully.")


# 1.3 Helper functions (Kaggle-specific)
# Note: These are only needed if running in Kaggle Notebooks environment
# For local use, you can skip these or adapt them as needed

def get_adk_proxy_url():
    """
    Gets the proxied URL in the Kaggle Notebooks environment.
    Only needed for Kaggle Notebooks - not required for local execution.
    """
    try:
        from IPython.core.display import display, HTML
        from jupyter_server.serverapp import list_running_servers
        
        PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
        ADK_PORT = "8000"

        servers = list(list_running_servers())
        if not servers:
            raise Exception("No running Jupyter servers found.")

        baseURL = servers[0]['base_url']

        try:
            path_parts = baseURL.split('/')
            kernel = path_parts[2]
            token = path_parts[3]
        except IndexError:
            raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

        url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
        url = f"{PROXY_HOST}{url_prefix}"

        styled_html = f"""
        <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
            <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
                <strong>⚠️ IMPORTANT: Action Required</strong>
            </div>
            <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
                The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
                <ol style="margin-top: 10px; padding-left: 20px;">
                    <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                    <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                    <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
                </ol>
                <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
            </div>
            <a href='{url}' target='_blank' style="
                display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
                text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
                Open ADK Web UI (after running cell below) ↗
            </a>
        </div>
        """

        display(HTML(styled_html))
        return url_prefix
    except ImportError:
        print("⚠️  Kaggle-specific helper function skipped (not in Kaggle environment)")
        return None

print("✅ Helper functions defined.")


# ============================================================================
# Section 2: Your first AI Agent with ADK
# ============================================================================

# 2.1 What is an AI Agent?
# 
# You've probably used an LLM like Gemini before, where you give it a prompt 
# and it gives you a text response.
# 
# Prompt -> LLM -> Text
# 
# An AI Agent takes this one step further. An agent can think, take actions, 
# and observe the results of those actions to give you a better answer.
# 
# Prompt -> Agent -> Thought -> Action -> Observation -> Final Answer
# 
# In this script, we'll build an agent that can take the action of searching Google.


# 2.2 Define your agent
# 
# We'll configure an Agent by setting its key properties:
# - name and description: A simple name and description to identify our agent
# - model: The specific LLM that will power the agent's reasoning (gemini-2.5-flash-lite)
# - instruction: The agent's guiding prompt
# - tools: A list of tools that the agent can use (google_search)

root_agent = Agent(
    name="helpful_assistant",
    model="gemini-2.5-flash-lite",
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

print("✅ Root Agent defined.")


# 2.3 Run your agent
# 
# To run the agent, we need a Runner, which is the central component within ADK 
# that acts as the orchestrator. It manages the conversation, sends our messages 
# to the agent, and handles its responses.

# Create an InMemoryRunner and tell it to use our root_agent
runner = InMemoryRunner(agent=root_agent)

print("✅ Runner created.")

# Note: You can also run agents using ADK command-line tools such as:
# - adk run
# - adk web
# - adk api_server


async def main():
    """Main function to run the agent examples."""
    
    # Example 1: Ask about ADK
    print("\n" + "="*70)
    print("Example 1: What is Agent Development Kit from Google?")
    print("="*70 + "\n")
    
    response = await runner.run_debug(
        "What is Agent Development Kit from Google? What languages is the SDK available in?"
    )
    
    # Example 2: Ask a question requiring current information
    print("\n" + "="*70)
    print("Example 2: What new movies are showing in theaters now in Los Angeles?")
    print("="*70 + "\n")
    
    response = await runner.run_debug(
        "What new movies are showing in theaters now in Los Angeles?"
    )
    
    # Your Turn! Try asking your own questions:
    # - What's the weather in London?
    # - Who won the last soccer world cup?
    # - What new movies are showing in theaters now?
    
    print("\n" + "="*70)
    print("✅ Agent examples completed!")
    print("="*70)
    print("\n💡 Try modifying the questions above or add your own!")


# ============================================================================
# Section 3: ADK Web Interface (Optional)
# ============================================================================

# To use the ADK web UI:
# 1. Create an agent with Python files using: adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY
# 2. Run the web server: adk web
# 3. Access the UI at http://127.0.0.1:8000
# 
# For Kaggle Notebooks, use: adk web --url_prefix {url_prefix}
# where url_prefix comes from get_adk_proxy_url()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

