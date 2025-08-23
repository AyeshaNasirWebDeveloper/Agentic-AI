from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

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

class BlogPostOutput(BaseModel):
    title: str
    introduction: str
    main_points: list[str]
    conclusion: str

async def main():
    agent = Agent(
        name = "BloggerAgent",
        instructions = "You are a helpful assistant that helps users write blog posts.",
        output_type = BlogPostOutput
)

    result = Runner.run_streamed(
        agent,
        input="Write a blog post about the benefits of using AI in everyday life.",
        run_config = config
)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())