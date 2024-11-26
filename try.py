import os
from openai import AsyncOpenAI
import asyncio
import json

client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
SEED = 11
async def CallGPT (messages, allow_functions=False):
    return await client.chat.completions.create(
        model='gpt-4o',
        seed=SEED,
        messages=messages,
        **({"tools": predefined_functions} if allow_functions else {})
    )

def create_coordinator_messages (m, pre_prompt, n, answers):
    answers_str = ""
    for a_i in range(len(answers)):
        iteration = answers[a_i]
        answers_str += f"\n\n\nIteration {a_i+1}:\n\n"
        for i in range(n):
            answers_str += f"Agent {i+1} answered: {iteration[i]}"

    return [
        {"role": "system", "content": "You are a coordinator for many different LLM agents."},
        {"role": "user", "content": f"This describes the task I want to complete: \n{pre_prompt}\n\nHistory so far:\n\n{answers_str}\n\n\n{m}"}
    ]

async def call_agent_iteration (pre_prompt, answers):
    answers_str = ""
    for a_i in range(len(answers)):
        iteration = answers[a_i]
        answers_str += f"\n\n\nIteration {a_i+1}:\n\n"
        for i in range(len(iteration)):
            answers_str += f"Agent {i+1} answered: {iteration[i]}"

    return (await CallGPT([
        {"role": "system", "content": "You are a coordinator for many different LLM agents."},
        {"role": "user", "content": f"This describes the task I want to complete: \n{pre_prompt}\n\nHistory so far:\n\n{answers_str}"}
    ], False)).choices[0].message.content

MAX_ITERATIONS = 10
async def hidden_state_agent (num_agents, pre_prompt):
    iteration = 1
    answers = []
    while iteration <= MAX_ITERATIONS:
        answer = []
        for _ in range(num_agents):
            answer.append(await call_agent_iteration(pre_prompt, answers))

        answers.append(answer)
        m = create_coordinator_messages("Have we completed execution? Answer yes if we have completed execution and we do not need another iteration? Answer yes/no.", pre_prompt, num_agents, answers)
        if "yes" in (await CallGPT(m)).choices[0].message.content.lower():
            break
        iteration += 1
    return answers

predefined_functions = [
    {
        "type": "function",
        "function": {
            "name": "hidden_state_agent",
            "description": """Launch multiple agents, but they write to some shared state one after another, so that the prior agent does not see what the next agent enters before it enters something.
Each agent does not see other agent's response in prompt!""",

            "parameters": {
                "type": "object",
                "properties": {
                    "num_agents": {
                        "type": "integer",
                        "description": "The number of agents to launch"
                    },
                    "pre_prompt": {
                        "type": "string",
                        "description": "The prompt to give to the first agent, carries forward"
                    }
                },
                "required": ["num_agents", "agent_pre_prompt"],
                "additionalProperties": False
            }
        }
    }
]

main_req = asyncio.run(CallGPT([{"role": "user", "content": """
Play rock paper scissors. Break into 2 agents, and tell them to select one between rock, paper and scissors. Tell them to only use one word.
Once you have a pick like rock paper or scissors from both, you can stop execution.
"""}], allow_functions=True))
print(main_req)

print(main_req.choices)
for choice in main_req.choices:
    if choice.finish_reason == "tool_calls":
        for tool_call in choice.message.tool_calls:
            f = tool_call.function
            if f.name == "hidden_state_agent":
                conversation = asyncio.run(hidden_state_agent(**json.loads(f.arguments)))

print("\n\nConversation Replay:")
for i in range(len(conversation)):
    print(f"\nIteration {i+1}:")
    for j in range(len(conversation[i])):
        print(f"Agent {j+1}: {conversation[i][j]}")
