from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool
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

class DeliveryOutput(BaseModel):
    address: str
    delivery_time: str
    items: list[str]

@function_tool
def schedule_delivery(address: str, delivery_time: str, items: list[str]) -> DeliveryOutput:
    return DeliveryOutput(
        address=address,
        delivery_time=delivery_time,
        items=items
    )

async def main():
    agent = Agent(
        name = "DeliveryAgent",
        instructions = "You are a helpful assistant that helps users schedule deliveries. Ask for the delivery address, preferred delivery time, and items to be delivered.",
        output_type = DeliveryOutput,
        tools = [schedule_delivery]
)

    result = Runner.run_streamed(
        agent,
        input="Schedule a delivery to 123 Main St at 5 PM with a pizza and a coldrink.",
        run_config = config
)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())