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

🚀 Memory Management - Part 1 - Sessions

Welcome to Day 3 of the Kaggle 5-day Agents course!

In this script, you'll learn:
- ✅ What sessions are and how to use them in your agent
- ✅ How to build stateful agents with sessions and events
- ✅ How to persist sessions in a database
- ✅ Context management practices such as context compaction
- ✅ Best practices for sharing session State
"""

import os
import asyncio
import sqlite3
from typing import Any, Dict

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
#     print("✅ Gemini API key setup complete.")
# except Exception as e:
#     print(f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")

# For local use, set the API key from environment variable
if "GOOGLE_API_KEY" not in os.environ:
    print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables.")
    print("   Please set it using: export GOOGLE_API_KEY='your-api-key'")
else:
    print("✅ Gemini API key setup complete.")


# 1.3: Import ADK components
from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.function_tool import FunctionTool
from google.genai import types

print("✅ ADK components imported successfully.")


# 1.4: Helper functions
# Helper function that manages a complete conversation session, handling session
# creation/retrieval, query processing, and response streaming. It supports
# both single queries and multiple queries in sequence.

# Global variables for session management
APP_NAME = "default"
USER_ID = "default"
MODEL_NAME = "gemini-2.5-flash-lite"
session_service = None  # Will be initialized later


async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    """Run a session with the agent.
    
    Args:
        runner_instance: The Runner instance to use
        user_queries: Single query string or list of queries
        session_name: Name of the session
    """
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query_content = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query_content
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("✅ Helper functions defined.")


# 1.5: Configure Retry Options
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# ============================================================================
# Section 2: Session Management
# ============================================================================

# 2.1 The Problem
# 
# At their core, Large Language Models are inherently stateless. Their awareness 
# is confined to the information you provide in a single API call. This means an 
# agent without proper context management will react to the current prompt without 
# considering any previous history.
# 
# Why does this matter? Imagine trying to have a meaningful conversation with 
# someone who forgets everything you've said after each sentence. That's the 
# challenge we face with raw LLMs!
# 
# In ADK, we use Sessions for short term memory management and Memory for long 
# term memory.

# 2.2 What is a Session?
# 
# A session is a container for conversations. It encapsulates the conversation 
# history in a chronological manner and also records all tool interactions and 
# responses for a single, continuous conversation. A session is tied to a user 
# and agent; it is not shared with other users.
# 
# In ADK, a Session is comprised of two key components Events and State:
# 
# Session.Events: Building blocks of a conversation
# - User Input: A message from the user (text, audio, image, etc.)
# - Agent Response: The agent's reply to the user
# - Tool Call: The agent's decision to use an external tool or API
# - Tool Output: The data returned from a tool call
# 
# Session.State: The Agent's scratchpad, where it stores and updates dynamic 
# details needed during the conversation. Think of it as a global {key, value} 
# pair storage which is available to all subagents and tools.

# 2.3 How to manage sessions?
# 
# An agentic application can have multiple users and each user may have multiple 
# sessions with the application. To manage these sessions and events, ADK offers 
# a Session Manager and Runner.
# 
# 1. SessionService: The storage layer
#    - Manages creation, storage, and retrieval of session data
#    - Different implementations for different needs (memory, database, cloud)
# 
# 2. Runner: The orchestration layer
#    - Manages the flow of information between user and agent
#    - Automatically maintains conversation history
#    - Handles the Context Engineering behind the scenes
# 
# Think of it like this:
# - Session = A notebook 📓
# - Events = Individual entries in a single page 📝
# - SessionService = The filing cabinet storing notebooks 🗄️
# - Runner = The assistant managing the conversation 🤖

# 2.4 Implementing Our First Stateful Agent
# 
# Let's build our first stateful agent that can remember and have constructive 
# conversations. We'll start with a simple Session Management option 
# (InMemorySessionService):


def setup_stateful_agent():
    """Set up a stateful agent with InMemorySessionService."""
    global session_service
    
    # Step 1: Create the LLM Agent
    root_agent = Agent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="A text chatbot",
    )

    # Step 2: Set up Session Management
    # InMemorySessionService stores conversations in RAM (temporary)
    session_service = InMemorySessionService()

    # Step 3: Create the Runner
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    print("✅ Stateful agent initialized!")
    print(f"   - Application: {APP_NAME}")
    print(f"   - User: {USER_ID}")
    print(f"   - Using: {session_service.__class__.__name__}")
    
    return runner


# ============================================================================
# Section 3: Persistent Sessions with DatabaseSessionService
# ============================================================================

# While InMemorySessionService is great for prototyping, real-world applications 
# need conversations to survive restarts, crashes, and deployments.
# 
# ADK provides different SessionService implementations:
# - InMemorySessionService: Development & Testing (Lost on restart)
# - DatabaseSessionService: Self-managed apps (Survives restarts)
# - Agent Engine Sessions: Production on GCP (Fully managed)


def setup_persistent_agent():
    """Set up a persistent agent with DatabaseSessionService."""
    global session_service
    
    # Step 1: Create the agent
    chatbot_agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="A text chatbot with persistent memory",
    )

    # Step 2: Switch to DatabaseSessionService
    # SQLite database will be created automatically
    db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
    session_service = DatabaseSessionService(db_url=db_url)

    # Step 3: Create a new runner with persistent storage
    runner = Runner(agent=chatbot_agent, app_name=APP_NAME, session_service=session_service)

    print("✅ Upgraded to persistent sessions!")
    print(f"   - Database: my_agent_data.db")
    print(f"   - Sessions will survive restarts!")
    
    return runner


def check_data_in_db():
    """Check data stored in the SQLite database."""
    with sqlite3.connect("my_agent_data.db") as connection:
        cursor = connection.cursor()
        result = cursor.execute(
            "select app_name, session_id, author, content from events"
        )
        print([_[0] for _ in result.description])
        for each in result.fetchall():
            print(each)


# ============================================================================
# Section 4: Context Compaction
# ============================================================================

# As events are stored in full in the session Database, this quickly adds up. 
# For a long, complex task, this list of events can become very large, leading 
# to slower performance and higher costs.
# 
# ADK's Context Compaction feature automatically reduces the context that's stored 
# in the Session by summarizing past events.


def setup_compaction_agent(chatbot_agent):
    """Set up an agent with context compaction enabled."""
    global session_service
    
    # Re-define our app with Events Compaction enabled
    research_app_compacting = App(
        name="research_app_compacting",
        root_agent=chatbot_agent,
        # This is the new part!
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=3,  # Trigger compaction every 3 invocations
            overlap_size=1,  # Keep 1 previous turn for context
        ),
    )

    db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
    session_service = DatabaseSessionService(db_url=db_url)

    # Create a new runner for our upgraded app
    research_runner_compacting = Runner(
        app=research_app_compacting, session_service=session_service
    )

    print("✅ Research App upgraded with Events Compaction!")
    
    return research_runner_compacting


async def verify_compaction(session_service_instance, runner_instance):
    """Verify that compaction occurred in the session."""
    # Get the final session state
    final_session = await session_service_instance.get_session(
        app_name=runner_instance.app_name,
        user_id=USER_ID,
        session_id="compaction_demo",
    )

    print("--- Searching for Compaction Summary Event ---")
    found_summary = False
    for event in final_session.events:
        # Compaction events have a 'compaction' attribute
        if event.actions and event.actions.compaction:
            print("\n✅ SUCCESS! Found the Compaction Event:")
            print(f"  Author: {event.author}")
            print(f"\n Compacted information: {event}")
            found_summary = True
            break

    if not found_summary:
        print(
            "\n❌ No compaction event found. Try increasing the number of turns in the demo."
        )


# ============================================================================
# Section 5: Working with Session State
# ============================================================================

# Let's explore how to manually manage session state through custom tools. In this 
# example, we'll identify a transferable characteristic, like a user's name and 
# their country, and create tools to capture and save it.

# Define scope levels for state keys (following best practices)
USER_NAME_SCOPE_LEVELS = ("temp", "user", "app")


def save_userinfo(
    tool_context: ToolContext, user_name: str, country: str
) -> Dict[str, Any]:
    """
    Tool to record and save user name and country in session state.

    Args:
        tool_context: ADK-provided tool context
        user_name: The username to store in session state
        country: The name of the user's country
    """
    # Write to session state using the 'user:' prefix for user data
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country

    return {"status": "success"}


def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user name and country from session state.
    """
    # Read from session state
    user_name = tool_context.state.get("user:name", "Username not found")
    country = tool_context.state.get("user:country", "Country not found")

    return {"status": "success", "user_name": user_name, "country": country}


def setup_session_state_agent():
    """Set up an agent with session state management tools."""
    global session_service
    
    # Create an agent with session state tools
    root_agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="""A text chatbot.
    Tools for managing user context:
    * To record username and country when provided use `save_userinfo` tool. 
    * To fetch username and country when required use `retrieve_userinfo` tool.
    """,
        tools=[FunctionTool(func=save_userinfo), FunctionTool(func=retrieve_userinfo)],
    )

    # Set up session service and runner
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name="default")

    print("✅ Agent with session state tools initialized!")
    
    return runner


async def main():
    """Main function to run examples."""
    
    print("\n" + "="*70)
    print("Section 2: Testing Stateful Agent")
    print("="*70)
    
    # Section 2: Stateful Agent
    runner = setup_stateful_agent()
    
    # Test the stateful agent
    await run_session(
        runner,
        [
            "Hi, I am Sam! What is the capital of United States?",
            "Hello! What is my name?",  # This time, the agent should remember!
        ],
        "stateful-agentic-session",
    )
    
    print("\n" + "="*70)
    print("Section 3: Testing Persistent Sessions")
    print("="*70)
    
    # Section 3: Persistent Sessions
    runner = setup_persistent_agent()
    
    await run_session(
        runner,
        ["Hi, I am Sam! What is the capital of the United States?", "Hello! What is my name?"],
        "test-db-session-01",
    )
    
    # Check database contents
    print("\n--- Database Contents ---")
    check_data_in_db()
    
    print("\n" + "="*70)
    print("Section 4: Testing Context Compaction")
    print("="*70)
    
    # Section 4: Context Compaction
    chatbot_agent = LlmAgent(
        model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
        name="text_chat_bot",
        description="A text chatbot with persistent memory",
    )
    
    research_runner = setup_compaction_agent(chatbot_agent)
    
    # Turn 1
    await run_session(
        research_runner,
        "What is the latest news about AI in healthcare?",
        "compaction_demo",
    )

    # Turn 2
    await run_session(
        research_runner,
        "Are there any new developments in drug discovery?",
        "compaction_demo",
    )

    # Turn 3 - Compaction should trigger after this turn!
    await run_session(
        research_runner,
        "Tell me more about the second development you found.",
        "compaction_demo",
    )

    # Turn 4
    await run_session(
        research_runner,
        "Who are the main companies involved in that?",
        "compaction_demo",
    )
    
    # Verify compaction
    await verify_compaction(session_service, research_runner)
    
    print("\n" + "="*70)
    print("Section 5: Testing Session State")
    print("="*70)
    
    # Section 5: Session State
    runner = setup_session_state_agent()
    
    # Test conversation demonstrating session state
    await run_session(
        runner,
        [
            "Hi there, how are you doing today? What is my name?",  # Agent shouldn't know the name yet
            "My name is Sam. I'm from Poland.",  # Provide name - agent should save it
            "What is my name? Which country am I from?",  # Agent should recall from session state
        ],
        "state-demo-session",
    )
    
    # Inspect session state
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="state-demo-session"
    )

    print("\nSession State Contents:")
    print(session.state)
    print("\n🔍 Notice the 'user:name' and 'user:country' keys storing our data!")
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

