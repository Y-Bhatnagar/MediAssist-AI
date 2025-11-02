import gradio as gr
import asyncio, requests
import websockets
import httpx
import mimetypes

#defining the incoming and outgoing queues
incoming = asyncio.Queue()
outgoing = asyncio.Queue()
client_id = "frontend"

#history of messages
#history = []

#defining connection
async def backend_connection():
    async with websockets.connect(f"ws://127.0.0.1:8000/conversation/{client_id}") as ws:
        receiver = asyncio.create_task(receiving_msg(ws))
        sender = asyncio.create_task(sending_msg(ws))
        await asyncio.gather(receiver,sender)

#defining the background task of receiving message
async def receiving_msg(ws):
    while True:
        msg = await ws.recv()
        print(f"message received from backend: {msg}")
        await incoming.put(msg)

#defining the backgroud task to send message to backend
async def sending_msg(ws):
    while True:
        msg = await outgoing.get()
        await ws.send(msg)
        print(f"message send to backend: {msg}")

#defining the polling function
def poll_incoming (history):
    msgs = []
    try:
        while True:
            msg = incoming.get_nowait()
            print (f"message received from the backend queue: {msg}")
            msgs.append(msg)
    except asyncio.QueueEmpty:
        pass
    
    if msgs:
        for m in msgs:
            history.append({"role":"assistant", "content": m})
            print(f"message appended to history")
        return history
    return gr.update()
#def output_msg(message, history):
    #msg_to_backend = requests.post("http://127.0.0.1:8000/user_input", data = {"user_message" : f"{message}"})
    #history.append(("user", message))
    #return history

#function to send response to the backend
async def output_msg(msg, history):
    history.append({"role": "user", "content" : msg})
    await outgoing.put(msg)
    return "", history    

#function to receive input from backend
#async def input_msg(chatbot):
    #msg = await incoming.get()
    #history.append({"role": "assistant", "content" : msg})
    #return history


#def get_first_response():
    #backend_response = requests.get("http://127.0.0.1:8000/display_message")
    #msg = backend_response.json()["message"]
    #if msg and (not history or history[-1][1] != msg):
        #history.append(("system", msg))
        #return history, gr.update(value = 0)
    #return history, gr.update(value = 1)

#function to upload files
async def upload_files(files, history):
    file_upload = []
    
    for f in files:
        file_path = getattr (f,"name",f)
        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        with open(file_path, "rb") as file_data:
            content = file_data.read()
            file_upload.append(("files", (file_path, content, mime_type)))

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post("http://127.0.0.1:8000/analyse", files = file_upload)
    
    if response.status_code == 200:
        msg = response.json()["status"]
    else:
        msg = f"Upload failed with status code {response.status_code}."

    history.append({"role":"system", "content": msg})
    return history


with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# 🩺 Aayu, a Medical Assistant"),
    chatbot = gr.Chatbot(height=500, type="messages", show_label=False)
    msg = gr.Textbox(placeholder = "Type your message here...")
    files = gr.UploadButton("Upload PDF or Image", file_types = [".pdf", "image"], file_count = "multiple")
    
    msg.submit(output_msg, inputs = [msg, chatbot], outputs = [msg,chatbot])
    files.upload(upload_files, inputs = [files,chatbot], outputs = chatbot)
    
    # Launch background connection
    app.load(lambda: asyncio.run(backend_connection()))
    timer = gr.Timer(1.0)
    timer.tick(poll_incoming, chatbot, chatbot)
   
app.launch()