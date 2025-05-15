from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

# load all env as variables here
load_dotenv()

# using the cohere llm chat model cause its free
# 20 requests per minute and 1000 request in a month
llm = ChatCohere()

# our schema
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tool_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)
raw_response = agent_executor.invoke({"query": "What is the capital of france?"})


try:
    structured_response = parser.parse(raw_response.get("output")["text"])
    print(structured_response.topic)
except Exception as e:
    print("ERROR PARSING RESPONSE\n", "Error == ", e, "\n", "Raw response == ", raw_response)

