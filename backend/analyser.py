from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
import os
import getpass
import asyncio,json
from asyncio import to_thread

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult

if not os.getenv("DI_endpoint"):
    os.environ["DI_endpoint"] = getpass.getpass("Enter your backend endpoint: ")
    endpoint = os.environ["DI_endpoint"]

if not os.getenv("DI_key"):
    os.environ["DI_key"] = getpass.getpass("Enter your backend API key: ")
    key = os.environ["DI_key"]

dic = DocumentIntelligenceClient(endpoint = os.environ["DI_endpoint"],credential = AzureKeyCredential(key))

#defining instance of FastAPI
app = FastAPI()

#storage of message from Aayu
aayu_queue = asyncio.Queue() #queue to send messages to aayu
frontend_queue = asyncio.Queue()

#defining routing table
route = {
    "aayu" : "frontend",
    "frontend" : "aayu"
}

#defining the backend connection
class ConnectionManager():
    def __init__ (self):
        self.active_connections: dict[str, WebSocket] = {}

    #defining methods associted with this class
    async def connect (self, websocket: WebSocket, client_id : str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print (f"Client connected. Total number of connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id : str):
        self.active_connections.pop(client_id, None)
        print (f" Client disconnected. \n Total number of active connections: {len(self.active_connections)}")
    
    async def receive_message(self, client_id: str):
        while True:
            message = await self.active_connections[client_id].receive_text()
            print (f"received message from {client_id}: {message}")
            if client_id == "aayu":
                await frontend_queue.put(message)
            else:
                await aayu_queue.put(message)
    
    async def send_message(self, client_id : str):
        while True:
            if client_id == "aayu":
                message_frontend = await frontend_queue.get()
                target = route[client_id]
                print(f"sending message to {target}: {message_frontend}")
                await self.active_connections[target].send_text(message_frontend)
            elif client_id == "frontend":
                message_aayu = await aayu_queue.get()
                target = route[client_id]
                print(f"sending message to {target}: {message_aayu}")
                await self.active_connections[target].send_text(message_aayu)

manager = ConnectionManager()
 
@app.websocket ("/conversation/{client_id}")
async def websocket_connection(websocket: WebSocket, client_id: str):
    await manager.connect(websocket,client_id)
    receiving = asyncio.create_task(manager.receive_message(client_id))
    sending = asyncio.create_task(manager.send_message(client_id))
    try:
        await asyncio.wait([receiving,sending], return_when = asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    finally:
        receiving.cancel()
        sending.cancel()

#defining the path operation for the file upload
@app.post("/analyse")
async def upload_files (files: list[UploadFile] = File(...)):
    uploaded_files = {}

    for file in files:
        contents = await file.read()
        # uploading the saved file to azure document intelligence for analysis
        result = await to_thread (lambda: dic.begin_analyze_document("prebuilt-layout",body = contents).result())
        uploaded_files[f"{file.filename}"] = str(result.content)
    
    message = json.dumps(uploaded_files)
    await aayu_queue.put(message)
    return {"status" : "Analysing the files...."}