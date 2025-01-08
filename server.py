import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# import asyncio
# from model import ModelProcessor

app = FastAPI()

# Initialize the model processor
# model_processor = ModelProcessor()


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
            # Receive audio data from client
            data = await websocket.receive_bytes()
            # Process audio data
            # processed_data = await asyncio.get_event_loop().run_in_executor(
            #     None, model_processor.process_audio_chunk, data
            # )

            import numpy as np

            processed_data = np.flip(data)

            if processed_data:
                # Send processed audio back to client
                await websocket.send_bytes(processed_data)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
