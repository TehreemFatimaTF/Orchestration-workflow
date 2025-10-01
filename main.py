# 🌐 Multi-Agent Workflow with Gemini
from agents import (
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    RunConfig,
    Runner,
    Agent,
    set_tracing_disabled,
    enable_verbose_stdout_logging
)
import os
from dotenv import load_dotenv

# 🔐 Load environment variables
load_dotenv()
enable_verbose_stdout_logging()

# 🌟 Setup Gemini client
gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 🧠 Define model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# ⚙️ Run configuration
config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# 🤖 Define Agents with enhanced instructions
web_search_agent = Agent(
    name="WebSearchAgent",
    instructions="""
    You are a professional Web Search Agent.
    - Search and summarize information from the web.
    - Provide accurate, up-to-date, and concise results.
    - Focus on clarity and reliability of sources.
    """,
)

data_analysis_agent = Agent(
    name="DataAnalysisAgent",
    instructions="""
    You are a skilled Data Analysis Agent.
    - Analyze the given information carefully.
    - Extract key insights, trends, and patterns.
    - Present findings in a structured and easy-to-understand way.
    """,
)

writer_agent = Agent(
    name="WriterAgent",
    instructions="""
    You are a Creative Writer Agent.
    - Convert analyzed insights into a polished, engaging explanation.
    - Write in a professional, clear, and reader-friendly style.
    - Ensure the content is smooth, compelling, and easy to read.
    """,
)

# 🚀 Agent Execution Pipeline
web_search_output = Runner.run_sync(
    web_search_agent,
    input="Tell me about LLM (Large Language Models).",
    run_config=config,
)

data_analysis_output = Runner.run_sync(
    data_analysis_agent,
    input=f"Analyze this data: {web_search_output.final_output}",
    run_config=config,
)

writer_output = Runner.run_sync(
    writer_agent,
    input=f"Write a clear explanation based on this analysis: {data_analysis_output.final_output}",
    run_config=config,
)

# 📝 Final Output
print("\n📌 Final Generated Content:\n")
print(writer_output.final_output)
