# since we wil be using the LLm interference we will be using the package langchain-azure-ai from https://python.langchain.com/docs/integrations/providers/microsoft/
import getpass #The getpass module is used to safely input passwords or sensitive information from the user without displaying them on the screen (no echo).
import os # allows to interact with the operating system
import asyncio # provides support for asynchronous programming
import websockets # used for creating WebSocket connections
import re, uuid
from datetime import datetime
from langchain_community.document_loaders import TextLoader #document loader 
from langchain_text_splitters import RecursiveCharacterTextSplitter #text_splitter

#checking if the key is available in envornment
if not os.getenv("AZURE_INFERENCE_CREDENTIAL"):
    os.environ ["AZURE_INFERENCE_CREDENTIAL"]= getpass.getpass("Enter your Azure API key: ")

#ckecking if the endpoint is available in the enviornment
if not os.getenv("AZURE_INFERENCE_ENDPOINT"):
    os.environ ["AZURE_INFERENCE_ENDPOINT"]= getpass.getpass("Enter endpoint: ").strip()

#checking if the embedding service endpoint is available in the enviornment
if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = getpass.getpass("Enter the Embedding Model endpoint : ")

#checking if the vector stores endpoint and API is available
if not os.getenv("Azure_search_endpoint"):
    os.environ["Azure_search_endpoint"] = getpass.getpass("Enter the Azure Search Endpoint: ")

if not os.getenv("Azure_search_API"):
    os.environ["Azure_search_API"] = getpass.getpass("Enter the Azure Search API: ")

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
# embedding model
from langchain_openai import AzureOpenAIEmbeddings
embedding_engine = AzureOpenAIEmbeddings(
    azure_endpoint = os.environ ["AZURE_OPENAI_ENDPOINT"],
    api_key = os.environ ["AZURE_INFERENCE_CREDENTIAL"],
    model="text-embedding-3-large",
    api_version = "2024-02-01"
)

#Vector Store
from langchain_community.vectorstores.azuresearch import AzureSearch
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint = os.environ ["Azure_search_endpoint"],
    azure_search_key = os.environ ["Azure_search_API"],
    index_name = "patient_test",
    embedding_function = embedding_engine.embed_query
)

#reteriver
from langchain_community.retrievers import AzureAISearchRetriever
retriever = AzureAISearchRetriever(
    service_name = "aayu-vector-store",
    api_key = os.environ ["Azure_search_API"],
    content_key="content", top_k=5, index_name="patient_test"
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

#Defining Aayu's prompt template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Your name is Aayu, you are an medical professional. Your task is to have a friendly conversation with the user to gather information about their symptoms.
        You should ask relevant follow up questions to understand patients (Name, age), their medical history, their symptoms, their allergies, any medications they might be taking, or any other information
        that might be relevant to diagnose their condition. If you feel necessary, you could also request the user to kindly share any medical test report available with them using the upload button, and analyse it.
        Once you have determined that the user has shared adequate amount of information that could help a doctor diagnose their condition, please confirm with them if they would like to share
        any other detail. If they don't wish to share anything else, proceed to invoke the tool to summarise the conversation. Once, summary is ready, please guess which medical specialist (E.g: General Physician, Neurologist,
        dermatologist) should the user must consult, and inform them only if you are 100 per cent or more sure in guess. In end please invoke the state_writer node to save the conversation in a file.
        While replying do not repeat yourself, and never ask questions for which the user has already provided information. if at any point of time user requests to switch to doctor mode, then switch to Dr node"""),
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

#defining prompt for the dr mode
dr_mode_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Are an experienced Medical professional, whose goal is to assist doctor with the diagnosis using all the accessible means. First time the user connects to dr mode, welcome then and politely
        enquire about the assistace they may need. Whenever you give any medical advice, it is very crucial for you to include the source that you have refered to while replying to doctor.
        Some Scenarios you can follow:
        In case doctor wishes to reveiw a patient's case: Reterive patient files using the reteriving process
        If doctor enquires about some facts reterive the related docs and inform the doctor in case relevant information is not available inform the doctor.
        if doctor enquires about something else use retrival and other resources to answer to best of your capacity, but never ever hallucinate or create false information. 
        Reterving process: reterive the information using the gatherer_node, and compose a poilte, helpful reply"""),
        ("user", "{input}")
    ]
)

#defining investigator prompt
i_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Your task is to analyse the converstaion, then gather the relevant information using tools. Once a tool shares an information
        judge whether it has provided sufficient information. If the information is not sufficient, identify what is missing and use tools to reterive it. Once you have the complete
        information, compile it along with the list of sources that you have refered. in case you couldn't reterive information, simply inform what you were unable to find,
        but never generate reando answers not linked to source"""),
        ("user", "{input}")
    ]
)

#defining tools
from langchain.tools import tool
#defining the tool to retrive the files associated with the user query
@tool
async def reteriver (query: str):
    "Use this tool to retervive information about a patient from the vector store"
    docs = await retriever.ainvoke(query)
    print (f"retervied succesfully\n {docs}")
    #docs_2 = await vector_store.asimilarity_search(query = query,k=5,search_type="hybrid")
    #print (f"vector similarity search result: \n {docs_2}")
    result = "\n\n".join([doc.page_content for doc in docs])
    return result

#defining schema for the structured output from Aayu for patient
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_core.messages import AnyMessage
class summary_format (BaseModel):
    Patient_Name: Optional[str] = Field(None, description="Patient's Name")
    Age: Optional[str] = Field(None, description="Patient's age")
    Symptoms: Optional[str] = Field(None, description="The symptoms reported by the patient")
    Medical_history: Optional[str] = Field(None, description="The medical history of the patient")
    Allergies: Optional[str] = Field(None, description="Any allergies the patient has")
    Medications: Optional[str] = Field(None, description="Any medications the patient is currently taking")
    Highlights: Optional[str] = Field(None, description= "Any important facts that a doctor should know about the patient but wasn't covered above")
    Summary: Optional[str] = Field(None,description="A concise summary of the patient's condition and symptoms in sentences")

    #formating
    def __str__(self):
        lines = []
        for field, value in self.model_dump().items():
            lines.append(f"{field}: {value}")
        return "\n".join(lines)

class guide (BaseModel):
    response: list[AnyMessage] = Field(None, description="llm's response to users input")
    next_steps: Literal ["asking_node", "patient_summary_writer_node","state_writer","dr_node","END"] = Field(None, description="Decide the next step based on the conversation with the patient")

class dr_guide (BaseModel):
    response: list[AnyMessage] = Field(None, description="llm's response to users input")
    next_steps: Literal ["receiving_node","information_gatherer","END"] = Field (None, description="Decide the next step based on the conversation with the doctor" )

#Starting the graph
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

memory = InMemorySaver()
    
#defining the state
class conversation (MessagesState):
    pass

#attaching the schema and chaining the llm
llm_1 = assist.with_structured_output(guide)
chat = prompt | llm_1

llm_2 = patient_summariser.with_structured_output(summary_format)
agent_summariser = Summary_prompt | llm_2

llm_3 = assist.with_structured_output(dr_guide)
advice = dr_mode_prompt | llm_3

llm_4 = assist.bind_tools([reteriver])
gatherer = i_prompt | llm_4

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
    print (f"\naayu's response: {response_1}\n")
    print (f"\nnext node: {invocation.next_steps}\n")
    if response_1 == None:
        response_1 = AIMessage(content = f"{response_1}")
    else:
        response = response_1[0].content
        await send_queue.put(response)
    next_step = invocation.next_steps
    return Command (
        update = {"messages": response_1},
        goto = {next_step}
    )

#defining the node to write the Graph's state to file
async def state_writer(state: conversation):
    "Write the graph's state to a text file"
    identity = datetime.now().isoformat()

    #Building base folder
    base_folder = "patient_logs"
    os.makedirs(base_folder, exist_ok = True)

    #extracting summary
    summary = ""
    for msg in reversed(state["messages"]):
        if "Patient_Name" in msg.content:
            summary = msg.content.strip()
            match = re.search(r"Patient_Name:\s*(.+)", summary)
            if match:
                patient_name = match.group(1)
            else:
                patient_name = "Unknown"
            break
    
    #creating sub folder
    safe_name = re.sub(r'[^A-Za-z0-9 _-]', '', patient_name)
    patient_folder = os.path.join(base_folder, safe_name)
    os.makedirs(patient_folder, exist_ok=True)

    #extracting the conversation for embedding
    conversation = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            role = "Aayu"
        elif isinstance(msg, HumanMessage):
            role = patient_name
        else:
            role = "Unknown"
        conversation.append(f"{role}:{msg.content.strip()}")

    #making the file
    file_name = f"Visit_{identity}.txt"
    file_path = os.path.join(patient_folder, file_name)
    #writing to the file
    with open (file_path,'a+') as file:
        file.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
        file.write(f"{summary}\n")
        file.write("\n".join(conversation))
    
    msg = AIMessage(content = f"{file_name} created successfully at file path:{file_path}")
    return Command(
        update = {"messages": msg},
        goto = "aayu_vector_node")

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
        goto = "Aayu_node")

#defining the Aayu node to embed the user interaction and store in vector database
async def aayu_vector_node (state: conversation):
    
    #extracting the file path
    for msg in reversed(state["messages"]):
        if "file path" in msg.content:
            match = re.search(r"file path:\s*(.+)", msg.content.strip())
            if match:
                file_path = match.group(1)
            else:
                file_path = "Unknown"
            break

    from pathlib import Path
    p = Path(file_path)
    visit = p.name
    Patient_Name = p.parent.name
    print (f"file path is: {file_path}")

    #defining function to generate ID
    convo_id = str(uuid.uuid4()).replace("-","")[:16]
    print (f"id: {convo_id}")
    
    loader = TextLoader(file_path) #loading conversation
    docs = loader.load()
    
    print (f"document loader worked correctly. {len(docs)} created\n")
    print (f"{docs[0].page_content}\n")
    print (f"metadata {docs[0].metadata}\n")

    text_splitter = RecursiveCharacterTextSplitter (chunk_size= 350, chunk_overlap = 50, add_start_index=True, separators=["\nAayu:", "\n", ". ", " "])
    split_docs = text_splitter.split_documents(docs) #spliting documents
    print (f"text splitter succeceded. {len(split_docs)}\n")
    
    await vector_store.aadd_documents(documents = split_docs)
    
    print (f"vector stored")

    msg = AIMessage(content = "embedding added to the vector store successfully!")
    return Command(
        update = {"messages" : msg},
        goto = END
    )

#defining the dr. Aayu node
async def dr_node (state:conversation):
    "Opens doctor or dr mode for the doctor"
    msg = state["messages"]
    invocation = await advice.ainvoke({"input" : msg})
    ai_msgs = invocation.response
    print (f"\n DR_node response : {ai_msgs}")
    step = invocation.next_steps
    print (f"\n next step : {step}")
    if ai_msgs == None:
        ai_msg = AIMessage(content = f"{ai_msgs}")
    else:
        ai_msg = ai_msgs[0].content
        print(f"\n Number of messages : {len(ai_msgs)}")
        await send_queue.put(ai_msg)
    return Command (
        update = {"messages" : ai_msg},
        goto = {step}
    )

#defining the receiving node to fetch response from backend and add to state
async def receiving_node (state: conversation):
    "fetches response from backend and add to state"
    msg = await receive_queue.get()
    user_msg = HumanMessage(content = f"{msg}")
    return Command(
        update = {"messages" : user_msg},
        goto = "dr_node")

#LLM to reterive the conversation and share the update
async def information_gatherer (state:conversation):
    "gathers relevant information on the basis of conversation"
    msgs = state["messages"]
    result = await gatherer.ainvoke ({"input" : msgs})
    print (f"\n gather's response : {result}")
    info = result.content
    print (f"\n message to backend {info}")
    await send_queue.put(info)
    return {"messages": result}

#Tool node for reteriver
async def reteriver_node (state: conversation):
    print ("reached reteriver node")
    msg = state["messages"][-1]
    result = []
    tool_call = msg.tool_calls
    for call in tool_call:
        if call["name"] == "reteriver":
            observation = await reteriver.ainvoke(call["args"])
            result.append(ToolMessage(
                content = observation,
                tool_call_id=call["id"]
            ))
    return {"messages" : result}

#routing function
def route_tool (state : conversation):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
         n = last_msg.tool_calls
         print (f"\n routing function value: {n[0]['name']}_node")
         return f"{n[0]['name']}_node"
    
    return "dr_node"

graph = StateGraph(conversation)
graph.add_node("asking_node", asking_node)
graph.add_node("Aayu_node", Aayu_node)
graph.add_node("patient_summary_writer_node", patient_summary_writer_node)
graph.add_node("state_writer", state_writer)
graph.add_node("aayu_vector_node", aayu_vector_node)
graph.add_node("dr_node", dr_node)
graph.add_node("receiving_node", receiving_node)
graph.add_node("information_gatherer", information_gatherer)
graph.add_node("reteriver_node", reteriver_node)

graph.add_edge(START, "asking_node")
graph.add_conditional_edges("asking_node", route)
graph.add_conditional_edges("information_gatherer", route_tool)
graph.add_edge("reteriver_node", "information_gatherer")

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