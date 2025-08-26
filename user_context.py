import os
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool, RunContextWrapper
from dotenv import load_dotenv
from dataclasses import dataclass
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

@dataclass
class UserContext:
    user_id: int
    name: str
    email: str
    phone: str
    address: str

async def get_user_context_func(wrapper: RunContextWrapper[UserContext]) -> str:
    user_context = wrapper.context
    return f"User Context:\nName: {user_context.name}\nEmail: {user_context.email}\nPhone: {user_context.phone}\nAddress: {user_context.address}"

get_user_context_tool = function_tool(get_user_context_func)

@function_tool
def schedule_delivery(address: str, delivery_time: str, items: list[str]) -> DeliveryOutput:
    return DeliveryOutput(
        address=address,
        delivery_time=delivery_time,
        items=items
    )

async def main():
    user_data = UserContext(
        user_id=1,
        name="Ayesha",
        email="ayesha@gmail.com",
        phone="123-456-7890",
        address="123 Main St Apt 4B New York, NY 10001"
    )

    user_context_str = await get_user_context_func(RunContextWrapper(user_data))
    print("User Context:", user_context_str)

    agent = Agent[UserContext](
        name = "DeliveryAgent",
        instructions = (
            "You are a helpful assistant that helps users schedule deliveries. "
            "You always take into account the user's context (name, email, phone, address) "
            "from the provided UserContext. "
            "Use this context when confirming or scheduling deliveries, instead of asking again."
        ),
        output_type = DeliveryOutput,
        tools = [schedule_delivery, get_user_context_tool]
    )

    result = Runner.run_streamed(
        agent,
        input="Schedule a delivery at 5 PM with a pizza and a coldrink.",
        context=user_data,
        run_config = config
    )
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())