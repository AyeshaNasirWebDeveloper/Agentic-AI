from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)

agent = Agent(
        name = "SocialMediaPostAgent",
        instructions="You are a helpful assistant that helps users create engaging social media posts. First, ask for the topic and target audience. Then, generate a catchy post with relevant hashtags.",
)

result = Runner.run_sync(
        agent,
        input = "I want to create a post about the benefits of meditation for busy professionals.",
        run_config = config
)

print(result.final_output)