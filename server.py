import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# import numpy as np
# import librosa

app = FastAPI()


# Serve index.html at the root path
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


# Mount static files at the '/static' path
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            processed_data = await websocket.receive_bytes()
            if processed_data:
                await websocket.send_bytes(processed_data)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
