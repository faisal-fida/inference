import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import argparse
import numpy as np
import torch
from modules.commons import str2bool
from realtime import VoiceChanger, load_models


parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint-path", type=str, default=None, help="Path to the model checkpoint"
)
parser.add_argument("--config-path", type=str, default=None, help="Path to the vocoder checkpoint")
parser.add_argument(
    "--reference-audio-path", type=str, default="examples/reference/azuma_0.wav", help="Path to the reference audio"
)
parser.add_argument(
    "--fp16", type=str2bool, nargs="?", const=True, help="Whether to use fp16", default=True
)
parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
args = parser.parse_args()
cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"
device = torch.device(cuda_target if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} to Load models")
model_set = load_models(args)

config = {
    "block_time": 0.25,  # in seconds
    "crossfade_time": 0.05,
    "extra_time_ce": 2.5,
    "extra_time": 0.5,
    "extra_time_right": 2.0,
    "inference_cfg_rate": 0.7,
    "max_prompt_length": 3.0,
    "diffusion_steps": 10,
}

voice_changer = VoiceChanger(model_set, args.reference_audio_path, device, config)


app = FastAPI()


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    SAMPLE_RATE = 44100
    SAMPLES_PER_SECOND = SAMPLE_RATE
    
    audio_buffer = np.array([], dtype=np.float32)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"Received audio chunk size: {len(data)}")
            chunk = np.frombuffer(data, dtype=np.float32)
            
            audio_buffer = np.concatenate([audio_buffer, chunk])
            
            if len(audio_buffer) >= SAMPLES_PER_SECOND:
                to_process = audio_buffer[:SAMPLES_PER_SECOND]
                processed_data = voice_changer.process_audio_chunk(to_process)
                print(f"Processed audio chunk size: {len(processed_data)}")
                if processed_data is not None:
                    await websocket.send_bytes(processed_data.tobytes())
                
                audio_buffer = audio_buffer[SAMPLES_PER_SECOND:]

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")