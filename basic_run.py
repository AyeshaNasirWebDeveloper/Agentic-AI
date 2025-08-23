from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
import asyncio

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

async def main():
    agent = Agent(
        name = "FlightBookingAgent",
        instructions="You are a helpful assistant that helps users book flights. First ask for the destination, dates, and number of passengers. Then provide a list of available flights with prices. Finally, confirm the booking details with the user.",
)

    result = await Runner.run(
        agent,
        input="I want to book a flight to Dubai 15th of next month for 2 people.",
        run_config = config
)

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())