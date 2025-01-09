import torch
import numpy as np
import yaml
import librosa
import torchaudio
from dataclasses import dataclass
from hf_utils import load_custom_model_from_hf
from modules.commons import load_checkpoint, build_model, recursive_munch

@dataclass
class AudioConfig:
    samplerate: int = 44100
    block_time: float = 0.25  # seconds
    crossfade_time: float = 0.05
    extra_time_ce: float = 2.5
    extra_time: float = 0.5
    extra_time_right: float = 2.0
    diffusion_steps: int = 10
    inference_cfg_rate: float = 0.7
    max_prompt_length: float = 3.0

def setup_mel_converter(self, spect_params):
        self.to_mel = lambda x: torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.samplerate,
            n_fft=spect_params['n_fft'],
            win_length=spect_params['win_length'],
            hop_length=spect_params['hop_length'],
            n_mels=spect_params['n_mels'],
            f_min=spect_params.get('fmin', 0),
            f_max=None if spect_params.get('fmax', "None") == "None" else 8000,
            center=False
        ).to(self.device)(x)

class VoiceConverter:
    def __init__(self, device: str, fp16: bool = True):
        self.device = device
        self.fp16 = fp16
        self.config = AudioConfig()
        self.model_set = None
        self.prompt_condition = None
        self.mel2 = None 
        self.style2 = None
        self.reference_wav_name = ""
        self.vad_model = None
        self.vad_cache = {}
        self.campplus_model = None
        self.to_mel = None
        
        # Define frame sizes
        self.extra_frame = int(np.round(self.config.extra_time_ce * self.config.samplerate / (self.config.samplerate // 50))) * (self.config.samplerate // 50)
        self.extra_frame_right = int(np.round(self.config.extra_time_right * self.config.samplerate / (self.config.samplerate // 50))) * (self.config.samplerate // 50)
        self.sola_search_frame = self.config.samplerate // 50
        self.sola_buffer_frame = None
        
        self.setup_audio_processing()
        
    def setup_audio_processing(self):
        self.zc = self.config.samplerate // 50
        self.block_frame = int(np.round(self.config.block_time * self.config.samplerate / self.zc)) * self.zc
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(self.config.crossfade_time * self.config.samplerate / self.zc)) * self.zc
        self.setup_buffers()
        
    def setup_buffers(self):
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame + self.extra_frame_right,
            device=self.device, dtype=torch.float32
        )
        self.input_wav_res = torch.zeros(320 * self.input_wav.shape[0] // self.zc, 
                                       device=self.device, dtype=torch.float32)
        self.setup_resampler()
        
    def load_models(self):
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC", "DiT_uvit_tat_xlsr_ema.pth", "config_dit_mel_seed_uvit_xlsr_tiny.yml")
        
        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = 'DiT'
        model = build_model(model_params, stage="DiT")
        
        model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path, 
                                       load_only_params=True, ignore_modules=[], is_distributed=False)
        
        for key in model:
            model[key].eval()
            model[key].to(self.device)
            
        # Load CAMPPlus model
        from modules.campplus.DTDNN import CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model.eval()
        self.campplus_model.to(self.device)

        # Setup mel spectrogram conversion
        self.setup_mel_converter(config['preprocess_params']['spect_params'])
        
        self.model_set = (model, self.setup_semantic_fn(), self.setup_vocoder(model_params))
        
        from funasr import AutoModel
        self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

    async def process_audio_chunk(self, audio_data: bytes) -> bytes:
        # Convert bytes to numpy array
        input_array = np.frombuffer(audio_data, dtype=np.float32)
        
        # Process through VAD
        input_16k = librosa.resample(input_array, orig_sr=self.config.samplerate, target_sr=16000)
        vad_result = self.vad_model.generate(input=input_16k, cache=self.vad_cache, 
                                           is_final=False, chunk_size=1000 * self.config.block_time)
        
        if not vad_result[0]["value"]:
            return np.zeros_like(input_array).tobytes()
            
        # Update input buffers
        self.update_buffers(input_array)
        
        # Perform voice conversion
        converted = self.convert_voice()
        
        # Apply SOLA algorithm
        output = self.apply_sola(converted)
        
        return output.cpu().numpy().tobytes()

    def update_buffers(self, input_array: np.ndarray):
        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-input_array.shape[0]:] = torch.from_numpy(input_array).to(self.device)
        
        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()
        self.input_wav_res[-320 * (input_array.shape[0] // self.zc + 1):] = (
            self.resampler(self.input_wav[-input_array.shape[0] - 2 * self.zc:])[320:]
        )

    @torch.no_grad()
    def convert_voice(self) -> torch.Tensor:
        model, semantic_fn, vocoder_fn = self.model_set
        
        converted_waves_16k = self.input_wav_res
        S_alt = semantic_fn(converted_waves_16k.unsqueeze(0))
        
        ce_dit_frame_difference = int(self.config.extra_time_ce * 50)
        S_alt = S_alt[:, ce_dit_frame_difference:]
        
        target_lengths = torch.LongTensor([self.return_length]).to(S_alt.device)
        cond = model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        
        cat_condition = torch.cat([self.prompt_condition, cond], dim=1)
        
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.fp16 else torch.float32):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(self.mel2.device),
                self.mel2,
                self.style2,
                None,
                n_timesteps=self.config.diffusion_steps,
                inference_cfg_rate=self.config.inference_cfg_rate,
            )
            vc_target = vc_target[:, :, self.mel2.size(-1):]
            vc_wave = vocoder_fn(vc_target).squeeze()
            
        return vc_wave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
voice_converter = VoiceConverter(device)
voice_converter.load_models()
reference_path = "examples/reference/azuma_0.wav"

reference_wav, _ = librosa.load(reference_path, sr=voice_converter.config.samplerate)
voice_converter.reference_wav = reference_wav

# Update prompt condition
sr = voice_converter.config.samplerate
prompt_len = voice_converter.config.max_prompt_length
reference_wav = reference_wav[:int(sr * prompt_len)]
reference_wav_tensor = torch.from_numpy(reference_wav).to(voice_converter.device)

# Process for semantic features
ori_waves_16k = voice_converter.resampler(reference_wav_tensor.unsqueeze(0))
model, semantic_fn, _ = voice_converter.model_set
S_ori = semantic_fn(ori_waves_16k)

# Get speaker style
feat2 = torchaudio.compliance.kaldi.fbank(
    ori_waves_16k, 
    num_mel_bins=80,
    dither=0,
    sample_frequency=16000
)
feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
voice_converter.style2 = voice_converter.campplus_model(feat2.unsqueeze(0))

# Get mel spectrogram
voice_converter.mel2 = voice_converter.to_mel(reference_wav_tensor.unsqueeze(0))
target2_lengths = torch.LongTensor([voice_converter.mel2.size(2)]).to(voice_converter.mel2.device)

# Update prompt condition
voice_converter.prompt_condition = model.length_regulator(
    S_ori,
    ylens=target2_lengths,
    n_quantizers=3,
    f0=None
)[0]

voice_converter.reference_wav_name = reference_path


import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_bytes()
            processed_audio = await voice_converter.process_audio_chunk(message)
            await websocket.send_bytes(processed_audio)

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")