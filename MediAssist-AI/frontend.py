import gradio as gr
import asyncio, requests
import websockets
import httpx
import mimetypes

#defining the incoming and outgoing queues
incoming = asyncio.Queue()
outgoing = asyncio.Queue()
client_id = "frontend"

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

#function to send response to the backend
async def output_msg(msg, history):
    history.append({"role": "user", "content" : msg})
    await outgoing.put(msg)
    return "", history    

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


with gr.Blocks() as app:
    gr.HTML("""<style> body { background-color: #f7f7f7; font-family: sans-serif; } 
    #header {
            font-size: 60px;
            text-align: left;
            font-weight: bold;
            </style>""")
    gr.Markdown("### 🩺 Aayu, a Medical Assistant", elem_id="header"),
    chatbot = gr.Chatbot(height=500, show_label=False)
    with gr.Column():
        msg = gr.Textbox(placeholder = "Type your message here...", scale = 1)
        files = gr.File(label = "Upload PDF or Image", file_types = [".pdf", "image"], file_count = "multiple")
    
    msg.submit(output_msg, inputs = [msg, chatbot], outputs = [msg,chatbot])
    files.upload(upload_files, inputs = [files,chatbot], outputs = chatbot)
    
    # Launch background connection
    app.load(lambda: asyncio.run(backend_connection()))
    timer = gr.Timer(1.0)
    timer.tick(fn = poll_incoming, inputs = chatbot, outputs = chatbot)
   
app.launch()