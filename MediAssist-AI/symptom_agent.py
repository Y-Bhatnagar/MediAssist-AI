# since we wil be using the LLm interference we will be using the package langchain-azure-ai from https://python.langchain.com/docs/integrations/providers/microsoft/
import getpass #The getpass module is used to safely input passwords or sensitive information from the user without displaying them on the screen (no echo).
import os # allows to interact with the operating system
import asyncio # provides support for asynchronous programming
import websockets # used for creating WebSocket connections

#checking if the key is available in envornment
if not os.getenv("AZURE_INFERENCE_CREDENTIAL"):
    os.environ ["AZURE_INFERENCE_CREDENTIAL"]= getpass.getpass("Enter your Azure API key: ")

#ckecking if the endpoint is available in the enviornment
if not os.getenv("AZURE_INFERENCE_ENDPOINT"):
    os.environ ["AZURE_INFERENCE_ENDPOINT"]= getpass.getpass("Enter endpoint: ").strip()

#istantiation
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
assist = AzureAIChatCompletionsModel(
    endpoint = os.environ["AZURE_INFERENCE_ENDPOINT"],
    credential = os.environ ["AZURE_INFERENCE_CREDENTIAL"],
    model="grok-3-mini",
    temperature=0
)

patient_summariser = AzureAIChatCompletionsModel(
    endpoint = os.environ["AZURE_INFERENCE_ENDPOINT"],
    credential = os.environ ["AZURE_INFERENCE_CREDENTIAL"],
    model="grok-3-mini",
    temperature=0
)

# Defining the communication between Aayu and backend
send_queue = asyncio.Queue() #Message aayu sends to backend
receive_queue = asyncio.Queue() #message backend sends to aayu
client_id = "aayu"

#Backend connection as bc
async def backend_connection():
    async with websockets.connect (f"ws://127.0.0.1:8000/conversation/{client_id}") as bc:
        #Create sender and receiver tasks to run concurrently
        sender = asyncio.create_task(send_message(bc))
        receiver = asyncio.create_task(receive_message(bc))
        await asyncio.gather(sender,receiver)

#definining the asyncronous sender function
async def send_message(bc):
    while True:
        msg = await send_queue.get()
        print (f"message to send to backend: {msg}")
        await bc.send(msg)

#defining the asyncronous receiving function
async def receive_message(bc):
    while True:
        msg = await bc.recv()
        print (f"message received from the backend: {msg}")
        await receive_queue.put(msg)

#async def main():
    #await backend_connection()

#if __name__ == "__main__":
    #asyncio.run(main())

#Defining Aayu's prompt template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Your name is Aayu, you are an medical professional. Your task is to have a friendly conversation with the user to gather information about their symptoms.
        You should ask relevant follow up questions to understand patients (Name, age), their medical history, their symptoms, their allergies, any medications they might be taking, or any other information
        that might be relevant to diagnose their condition. 
        Once you have determined that the user has shared adequate amount of information that could help a doctor diagnose their condition, please confirm with them if they would like to share
        any other detail. If they don't wish to share anything else, proceed to invoke the tool to summarise the conversation.
        While replying do not repeat yourself, and never ask questions for which the user has already provided information."""),
        ("user", "{input}")
    ]
)

#defining prompt for the summary writer llm
Summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a medical professional. Your task is to analyse the conversation between Aayu and the patient and generate a concise summary of the patient's condition and symptoms.
        The summary should be easy to digest, should highlight any important facts that a doctor should know in a seperate paragraph. You should not lose any information while summarising the conversation."""),
        ("user", """Here is the conversation between Aayu and the patient: {patient_data}""")
    ]
)

#defining schema for the structured output from Aayu for patient
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_core.messages import AnyMessage
class summary_format (BaseModel):
    Patient_identification: Optional[list[str]] = Field(None, description="Patient's personal details like name, age and sex")
    Symptoms: Optional[str] = Field(None, description="The symptoms reported by the patient")
    Medical_history: Optional[str] = Field(None, description="The medical history of the patient")
    Allergies: Optional[str] = Field(None, description="Any allergies the patient has")
    Medications: Optional[str] = Field(None, description="Any medications the patient is currently taking")
    Highlights: Optional[str] = Field(None, description= "Any important facts that a doctor should know about the patient but wasn't covered above")
    summary: Optional[str] = Field(None,description="A concise summary of the patient's condition and symptoms in sentences")

    #formating
    def __str__(self):
        lines = []
        for field, value in self.model_dump().items():
            lines.append(f"{field}: {value}")
        return "\n".join(lines)

class guide (BaseModel):
    response: list[AnyMessage] = Field(None, description="Aayu's response to the patient")
    next_steps: Literal ["asking_node", "patient_summary_writer_node", "END"] = Field(None, description="Decide the next step based on the conversation with the patient")

#attaching the schema and chaining the llm
llm_1 = assist.with_structured_output(guide)
chat = prompt | llm_1

llm_2 = patient_summariser.with_structured_output(summary_format)
agent_summariser = Summary_prompt | llm_2

#defining schema for the structured output from Aayu for Doctor
"""class dr_summary_format (BaseModel):
    Patient_identification: str = Field(None, description="Patient's personal details like name, age and sex")
    Symptoms: str = Field(None, description="The symptoms reported by the patient")
    Medical_history: str = Field(None, description="The medical history of the patient")
    Allergies: str = Field(None, description="Any allergies the patient has")
    Medications: str = Field(None, description="Any medications the patient is currently taking")
    Diagonostic_1 : str = Field(None, description = "")
    Highlights: str = Field(None, description= "Any important facts that a doctor should know about the patient but wasn't covered above")"""

#Starting the graph
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
memory = InMemorySaver()
    
#defining the state
class conversation (MessagesState):
    pass

#defining nodes for graph
#defining the asking node: introduces Aayu and asks how the user is feeling
async def asking_node(state: conversation):
    """This nodes accepts and send messages from queue and appends it to the messages list in the state"""
    if len(state["messages"]) ==1:
        intro = state["messages"][-1].content
        await send_queue.put(intro)
        response = await receive_queue.get()
        input_msg = HumanMessage(content= str(response))
    else:
        response = await receive_queue.get()
        input_msg = HumanMessage(content= str(response))
    return {"messages": input_msg}

#defining the Aayu node: invokes llm and passes last message to it
async def Aayu_node(state: conversation):
    last_msg = state["messages"]
    invocation = await chat.ainvoke({"input": last_msg})
    response_1 = invocation.response
    response = response_1[0].content
    next_step = invocation.next_steps
    #print(f"\n Aayu: {response}\n")
    await send_queue.put(response)
    return Command (
        update = {"messages": response_1},
        goto = {next_step}
    )

#defining routing function
def route (state: conversation):
    last_message = state["messages"][-1].content
    if last_message.lower() == "done":
        return END
    else:
        return "Aayu_node"

#defining the patient summary writing node
async def patient_summary_writer_node(state: conversation):
    """This node will generate a summary of Aayu's conversation with the patient"""
    num_of_mssg = len(state["messages"])
    input_patient_summary_writer = "\n".join(f"{type(msg).__name__} : {msg.content}" for msg in state["messages"][0:(num_of_mssg - 1)])
    summariser = await agent_summariser.ainvoke({"patient_data": input_patient_summary_writer})
    #print(f"Summary of our conversation: \n{summariser}\n")
    await send_queue.put(str(summariser))
    msg = AIMessage(content = str(summariser)) # can use f string if the error persists
    return Command(
        update = {"messages": msg},
        goto = {"Aayu_node"})

graph = StateGraph(conversation)
graph.add_node("asking_node", asking_node)
graph.add_node("Aayu_node", Aayu_node)
graph.add_node("patient_summary_writer_node", patient_summary_writer_node)

graph.add_edge(START, "asking_node")
graph.add_conditional_edges("asking_node", route)

flow = graph.compile(checkpointer = memory)
config = {"configurable" : {"thread_id" : "1"}}
async def run():
    asyncio.create_task(backend_connection())
    await asyncio.sleep (1)
    start_msg = await flow.ainvoke({"messages": [AIMessage(content= "Hello! I'm Aayu. Please tell me how you are feeling?")]}, config)

if __name__ == "__main__":
    asyncio.run(run())

log = flow.get_state(config)
print (f"{log}\n")