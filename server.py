import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import argparse
import numpy as np
import torch
from modules.commons import str2bool
from realtime import VoiceChanger, load_models
import os
import time
from scipy.io import wavfile


SAMPLE_RATE = 44100
WAV_OUTPUT_DIR = "wav_output"
os.makedirs(WAV_OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint-path", type=str, default=None, help="Path to the model checkpoint"
)
parser.add_argument("--config-path", type=str, default=None, help="Path to the vocoder checkpoint")
parser.add_argument(
    "--reference-audio-path", type=str, required=True, help="Path to reference audio"
)
parser.add_argument(
    "--fp16", type=str2bool, nargs="?", const=True, help="Whether to use fp16", default=True
)
parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
args = parser.parse_args()
cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"
device = torch.device(cuda_target if torch.cuda.is_available() else "cpu")

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
    audio_chunks = []  # Buffer to store all chunks
    try:
        while True:
            processed_data = await websocket.receive_bytes()
            processed_data = np.frombuffer(processed_data, dtype=np.float32)
            processed_data = voice_changer.process_audio_chunk(processed_data)
            if processed_data is not None:
                processed_data = np.clip(processed_data, -1, 1)
                audio_chunks.append(processed_data)  # Store chunk in buffer
                await websocket.send_bytes(processed_data.tobytes())
    except WebSocketDisconnect:
        if audio_chunks:  # Save complete audio when connection ends
            complete_audio = np.concatenate(audio_chunks)
            audio_16bit = (complete_audio * 32767).astype(np.int16)
            timestamp = int(time.time())
            wav_filename = os.path.join(WAV_OUTPUT_DIR, f"complete_audio_{timestamp}.wav")
            wavfile.write(wav_filename, SAMPLE_RATE, audio_16bit)


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
