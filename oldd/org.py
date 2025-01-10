class GUIConfig:
    def __init__(self) -> None:
        self.reference_audio_path: str = ""
        # self.index_path: str = ""
        self.diffusion_steps: int = 10
        self.sr_type: str = "sr_model"
        self.block_time: float = 0.25  # s
        self.threhold: int = -60
        self.crossfade_time: float = 0.05
        self.extra_time_ce: float = 2.5
        self.extra_time: float = 0.5
        self.extra_time_right: float = 2.0
        self.I_noise_reduce: bool = False
        self.O_noise_reduce: bool = False
        self.inference_cfg_rate: float = 0.7
        self.sg_hostapi: str = ""
        self.wasapi_exclusive: bool = False
        self.sg_input_device: str = ""
        self.sg_output_device: str = ""


class GUI:
    def __init__(self, args) -> None:
        self.gui_config = GUIConfig()
        self.config = Config()
        self.function = "vc"
        self.delay_time = 0
        self.hostapis = None
        self.input_devices = None
        self.output_devices = None
        self.input_devices_indices = None
        self.output_devices_indices = None
        self.stream = None
        self.model_set = load_models(args)
        from funasr import AutoModel
        self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")
        self.update_devices()
        self.launcher()

    def load(self):
        try:
            os.makedirs("configs/inuse", exist_ok=True)
            if not os.path.exists("configs/inuse/config.json"):
                shutil.copy("configs/config.json", "configs/inuse/config.json")
            with open("configs/inuse/config.json", "r") as j:
                data = json.load(j)
                data["sr_model"] = data["sr_type"] == "sr_model"
                data["sr_device"] = data["sr_type"] == "sr_device"
                if data["sg_hostapi"] in self.hostapis:
                    self.update_devices(hostapi_name=data["sg_hostapi"])
                    if (
                        data["sg_input_device"] not in self.input_devices
                        or data["sg_output_device"] not in self.output_devices
                    ):
                        self.update_devices()
                        data["sg_hostapi"] = self.hostapis[0]
                        data["sg_input_device"] = self.input_devices[
                            self.input_devices_indices.index(sd.default.device[0])
                        ]
                        data["sg_output_device"] = self.output_devices[
                            self.output_devices_indices.index(sd.default.device[1])
                        ]
                else:
                    data["sg_hostapi"] = self.hostapis[0]
                    data["sg_input_device"] = self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ]
                    data["sg_output_device"] = self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ]
        except:
            with open("configs/inuse/config.json", "w") as j:
                data = {
                    "sg_hostapi": self.hostapis[0],
                    "sg_wasapi_exclusive": False,
                    "sg_input_device": self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ],
                    "sg_output_device": self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ],
                    "sr_type": "sr_model",
                    "block_time": 0.3,
                    "crossfade_length": 0.04,
                    "extra_time_ce": 2.5,
                    "extra_time": 0.5,
                    "extra_time_right": 0.02,
                    "diffusion_steps": 10,
                    "inference_cfg_rate": 0.7,
                    "max_prompt_length": 3.0,
                }
                data["sr_model"] = data["sr_type"] == "sr_model"
                data["sr_device"] = data["sr_type"] == "sr_device"
        return data

    def launcher(self):
        self.config = Config()
        data = self.load()
        sg.theme("LightBlue3")
        layout = [
            [
                sg.Frame(
                    title="Load reference audio",
                    layout=[
                        [
                            sg.Input(
                                default_text=data.get("reference_audio_path", ""),
                                key="reference_audio_path",
                            ),
                            sg.FileBrowse(
                                "choose an audio file",
                                initial_folder=os.path.join(
                                    os.getcwd(), "examples/reference"
                                ),
                                file_types=((". wav"), (". mp3"), (". flac"), (". m4a"), (". ogg"), (". opus")),
                            ),
                        ],
                    ],
                )
            ],
            [
                sg.Frame(
                    layout=[
                        [
                            sg.Text("Device type"),
                            sg.Combo(
                                self.hostapis,
                                key="sg_hostapi",
                                default_value=data.get("sg_hostapi", ""),
                                enable_events=True,
                                size=(20, 1),
                            ),
                            sg.Checkbox(
                                "WASAPI Exclusive Device",
                                key="sg_wasapi_exclusive",
                                default=data.get("sg_wasapi_exclusive", False),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Input Device"),
                            sg.Combo(
                                self.input_devices,
                                key="sg_input_device",
                                default_value=data.get("sg_input_device", ""),
                                enable_events=True,
                                size=(45, 1),
                            ),
                        ],
                        [
                            sg.Text("Output Device"),
                            sg.Combo(
                                self.output_devices,
                                key="sg_output_device",
                                default_value=data.get("sg_output_device", ""),
                                enable_events=True,
                                size=(45, 1),
                            ),
                        ],
                        [
                            sg.Button("Reload devices", key="reload_devices"),
                            sg.Radio(
                                "Use model sampling rate",
                                "sr_type",
                                key="sr_model",
                                default=data.get("sr_model", True),
                                enable_events=True,
                            ),
                            sg.Radio(
                                "Use device sampling rate",
                                "sr_type",
                                key="sr_device",
                                default=data.get("sr_device", False),
                                enable_events=True,
                            ),
                            sg.Text("Sampling rate:"),
                            sg.Text("", key="sr_stream"),
                        ],
                    ],
                    title="Sound Device",
                )
            ],
            [
                sg.Frame(
                    layout=[
                        # [
                        #     sg.Text("Activation threshold"),
                        #     sg.Slider(
                        #         range=(-60, 0),
                        #         key="threhold",
                        #         resolution=1,
                        #         orientation="h",
                        #         default_value=data.get("threhold", -60),
                        #         enable_events=True,
                        #     ),
                        # ],
                        [
                            sg.Text("Diffusion steps"),
                            sg.Slider(
                                range=(1, 30),
                                key="diffusion_steps",
                                resolution=1,
                                orientation="h",
                                default_value=data.get("diffusion_steps", 10),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Inference cfg rate"),
                            sg.Slider(
                                range=(0.0, 1.0),
                                key="inference_cfg_rate",
                                resolution=0.1,
                                orientation="h",
                                default_value=data.get("inference_cfg_rate", 0.7),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Max prompt length (s)"),
                            sg.Slider(
                                range=(1.0, 20.0),
                                key="max_prompt_length",
                                resolution=0.5,
                                orientation="h",
                                default_value=data.get("max_prompt_length", 3.0),
                                enable_events=True,
                            ),
                        ],
                    ],
                    title="Regular settings",
                ),
                sg.Frame(
                    layout=[
                        [
                            sg.Text("Block time"),
                            sg.Slider(
                                range=(0.04, 3.0),
                                key="block_time",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("block_time", 1.0),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Crossfade length"),
                            sg.Slider(
                                range=(0.02, 0.5),
                                key="crossfade_length",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("crossfade_length", 0.1),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Extra content encoder context time (left)"),
                            sg.Slider(
                                range=(0.5, 10.0),
                                key="extra_time_ce",
                                resolution=0.1,
                                orientation="h",
                                default_value=data.get("extra_time_ce", 5.0),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Extra DiT context time (left)"),
                            sg.Slider(
                                range=(0.5, 10.0),
                                key="extra_time",
                                resolution=0.1,
                                orientation="h",
                                default_value=data.get("extra_time", 5.0),
                                enable_events=True,
                            ),
                        ],
                        [
                            sg.Text("Extra context time (right)"),
                            sg.Slider(
                                range=(0.02, 10.0),
                                key="extra_time_right",
                                resolution=0.02,
                                orientation="h",
                                default_value=data.get("extra_time_right", 2.0),
                                enable_events=True,
                            ),
                        ],
                    ],
                    title="Performance settings",
                ),
            ],
            [
                sg.Button("Start Voice Conversion", key="start_vc"),
                sg.Button("Stop Voice Conversion", key="stop_vc"),
                sg.Radio(
                    "Input listening",
                    "function",
                    key="im",
                    default=False,
                    enable_events=True,
                ),
                sg.Radio(
                    "Voice Conversion",
                    "function",
                    key="vc",
                    default=True,
                    enable_events=True,
                ),
                sg.Text("Algorithm delay (ms):"),
                sg.Text("0", key="delay_time"),
                sg.Text("Inference time (ms):"),
                sg.Text("0", key="infer_time"),
            ],
        ]
        self.window = sg.Window("Seed-VC - GUI", layout=layout, finalize=True)
        self.event_handler()

    def event_handler(self):
        global flag_vc
        while True:
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED:
                self.stop_stream()
                exit()
            if event == "reload_devices" or event == "sg_hostapi":
                self.gui_config.sg_hostapi = values["sg_hostapi"]
                self.update_devices(hostapi_name=values["sg_hostapi"])
                if self.gui_config.sg_hostapi not in self.hostapis:
                    self.gui_config.sg_hostapi = self.hostapis[0]
                self.window["sg_hostapi"].Update(values=self.hostapis)
                self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)
                if (
                    self.gui_config.sg_input_device not in self.input_devices
                    and len(self.input_devices) > 0
                ):
                    self.gui_config.sg_input_device = self.input_devices[0]
                self.window["sg_input_device"].Update(values=self.input_devices)
                self.window["sg_input_device"].Update(
                    value=self.gui_config.sg_input_device
                )
                if self.gui_config.sg_output_device not in self.output_devices:
                    self.gui_config.sg_output_device = self.output_devices[0]
                self.window["sg_output_device"].Update(values=self.output_devices)
                self.window["sg_output_device"].Update(
                    value=self.gui_config.sg_output_device
                )
            if event == "start_vc" and not flag_vc:
                if self.set_values(values) == True:
                    printt("cuda_is_available: %s", torch.cuda.is_available())
                    self.start_vc()
                    settings = {
                        "reference_audio_path": values["reference_audio_path"],
                        # "index_path": values["index_path"],
                        "sg_hostapi": values["sg_hostapi"],
                        "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],
                        "sg_input_device": values["sg_input_device"],
                        "sg_output_device": values["sg_output_device"],
                        "sr_type": ["sr_model", "sr_device"][
                            [
                                values["sr_model"],
                                values["sr_device"],
                            ].index(True)
                        ],
                        # "threhold": values["threhold"],
                        "diffusion_steps": values["diffusion_steps"],
                        "inference_cfg_rate": values["inference_cfg_rate"],
                        "max_prompt_length": values["max_prompt_length"],
                        "block_time": values["block_time"],
                        "crossfade_length": values["crossfade_length"],
                        "extra_time_ce": values["extra_time_ce"],
                        "extra_time": values["extra_time"],
                        "extra_time_right": values["extra_time_right"],
                    }
                    with open("configs/inuse/config.json", "w") as j:
                        json.dump(settings, j)
                    if self.stream is not None:
                        self.delay_time = (
                            self.stream.latency[-1]
                            + values["block_time"]
                            + values["crossfade_length"]
                            + values["extra_time_right"]
                            + 0.01
                        )
                    self.window["sr_stream"].update(self.gui_config.samplerate)
                    self.window["delay_time"].update(
                        int(np.round(self.delay_time * 1000))
                    )
            # Parameter hot update
            # if event == "threhold":
            #     self.gui_config.threhold = values["threhold"]
            elif event == "diffusion_steps":
                self.gui_config.diffusion_steps = values["diffusion_steps"]
            elif event == "inference_cfg_rate":
                self.gui_config.inference_cfg_rate = values["inference_cfg_rate"]
            elif event in ["vc", "im"]:
                self.function = event
            elif event == "stop_vc" or event != "start_vc":
                # Other parameters do not support hot update
                self.stop_stream()

    def set_values(self, values):
        if len(values["reference_audio_path"].strip()) == 0:
            sg.popup("Choose an audio file")
            return False
        pattern = re.compile("[^\x00-\x7F]+")
        if pattern.findall(values["reference_audio_path"]):
            sg.popup("audio file path contains non-ascii characters")
            return False
        self.set_devices(values["sg_input_device"], values["sg_output_device"])
        self.gui_config.sg_hostapi = values["sg_hostapi"]
        self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]
        self.gui_config.sg_input_device = values["sg_input_device"]
        self.gui_config.sg_output_device = values["sg_output_device"]
        self.gui_config.reference_audio_path = values["reference_audio_path"]
        self.gui_config.sr_type = ["sr_model", "sr_device"][
            [
                values["sr_model"],
                values["sr_device"],
            ].index(True)
        ]
        # self.gui_config.threhold = values["threhold"]
        self.gui_config.diffusion_steps = values["diffusion_steps"]
        self.gui_config.inference_cfg_rate = values["inference_cfg_rate"]
        self.gui_config.max_prompt_length = values["max_prompt_length"]
        self.gui_config.block_time = values["block_time"]
        self.gui_config.crossfade_time = values["crossfade_length"]
        self.gui_config.extra_time_ce = values["extra_time_ce"]
        self.gui_config.extra_time = values["extra_time"]
        self.gui_config.extra_time_right = values["extra_time_right"]
        return True

    def start_vc(self):
        torch.cuda.empty_cache()
        self.reference_wav, _ = librosa.load(
            self.gui_config.reference_audio_path, sr=self.model_set[-1]["sampling_rate"]
        )
        self.gui_config.samplerate = (
            self.model_set[-1]["sampling_rate"]
            if self.gui_config.sr_type == "sr_model"
            else self.get_device_samplerate()
        )
        self.gui_config.channels = self.get_device_channels()
        self.zc = self.gui_config.samplerate // 50  # 44100 // 100 = 441
        self.block_frame = (
            int(
                np.round(
                    self.gui_config.block_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.block_frame_16k = 320 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(
                np.round(
                    self.gui_config.crossfade_time
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.gui_config.extra_time_ce
                    * self.gui_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.extra_frame_right = (
                int(
                    np.round(
                        self.gui_config.extra_time_right
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
        )
        self.input_wav: torch.Tensor = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
            + self.extra_frame_right,
            device=self.config.device,
            dtype=torch.float32,
        )  # 2 * 44100 + 0.08 * 44100 + 0.01 * 44100 + 0.25 * 44100
        self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
        self.input_wav_res: torch.Tensor = torch.zeros(
            320 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )  # input wave 44100 -> 16000
        self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer: torch.Tensor = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
        self.output_buffer: torch.Tensor = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.gui_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        if self.model_set[-1]["sampling_rate"] != self.gui_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.model_set[-1]["sampling_rate"],
                new_freq=self.gui_config.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        else:
            self.resampler2 = None
        self.vad_cache = {}
        self.vad_chunk_size = 1000 * self.gui_config.block_time
        self.vad_speech_detected = False
        self.set_speech_detected_false_at_end_flag = False
        self.start_stream()

    def start_stream(self):
        global flag_vc
        if not flag_vc:
            flag_vc = True
            if (
                "WASAPI" in self.gui_config.sg_hostapi
                and self.gui_config.sg_wasapi_exclusive
            ):
                extra_settings = sd.WasapiSettings(exclusive=True)
            else:
                extra_settings = None
            self.stream = sd.Stream(
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.gui_config.samplerate,
                channels=self.gui_config.channels,
                dtype="float32",
                extra_settings=extra_settings,
            )
            self.stream.start()

    def stop_stream(self):
        global flag_vc
        if flag_vc:
            flag_vc = False
            if self.stream is not None:
                self.stream.abort()
                self.stream.close()
                self.stream = None

    def audio_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        """
        Audio block callback function
        """
        global flag_vc
        print(indata.shape)
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)

        # VAD first
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        indata_16k = librosa.resample(indata, orig_sr=self.gui_config.samplerate, target_sr=16000)
        res = self.vad_model.generate(input=indata_16k, cache=self.vad_cache, is_final=False, chunk_size=self.vad_chunk_size)
        res_value = res[0]["value"]
        print(res_value)
        if len(res_value) % 2 == 1 and not self.vad_speech_detected:
            self.vad_speech_detected = True
        elif len(res_value) % 2 == 1 and self.vad_speech_detected:
            self.set_speech_detected_false_at_end_flag = True
        end_event.record()
        torch.cuda.synchronize()  # Wait for the events to be recorded!
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Time taken for VAD: {elapsed_time_ms}ms")

        self.input_wav[: -self.block_frame] = self.input_wav[
            self.block_frame :
        ].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
            self.config.device
        )
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1) :] = (
            self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[
                320:
            ]
        )
        print(f"preprocess time: {time.perf_counter() - start_time:.2f}")
        # infer
        if self.function == "vc":
            if self.gui_config.extra_time_ce - self.gui_config.extra_time < 0:
                raise ValueError("Content encoder extra context must be greater than DiT extra context!")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            infer_wav = custom_infer(
                self.model_set,
                self.reference_wav,
                self.gui_config.reference_audio_path,
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.skip_tail,
                self.return_length,
                int(self.gui_config.diffusion_steps),
                self.gui_config.inference_cfg_rate,
                self.gui_config.max_prompt_length,
                self.gui_config.extra_time_ce - self.gui_config.extra_time,
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)
            print(f"Time taken for VC: {elapsed_time_ms}ms")
            if not self.vad_speech_detected:
                infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])
        elif self.gui_config.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()

        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]

        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:

            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])

        print(f"sola_offset = {int(sola_offset)}")

        #post_process_start = time.perf_counter()
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += (
            self.sola_buffer * self.fade_out_window
        )
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        outdata[:] = (
            infer_wav[: self.block_frame]
            .repeat(self.gui_config.channels, 1)
            .t()
            .cpu()
            .numpy()
        )

        total_time = time.perf_counter() - start_time
        if flag_vc:
            self.window["infer_time"].update(int(total_time * 1000))

        if self.set_speech_detected_false_at_end_flag:
            self.vad_speech_detected = False
            self.set_speech_detected_false_at_end_flag = False

        print(f"Infer time: {total_time:.2f}")

    def update_devices(self, hostapi_name=None):
        """Get input and output devices."""
        global flag_vc
        flag_vc = False
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.hostapis = [hostapi["name"] for hostapi in hostapis]
        if hostapi_name not in self.hostapis:
            hostapi_name = self.hostapis[0]
        self.input_devices = [
            d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices = [
            d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]

    def set_devices(self, input_device, output_device):
        """set input and output devices."""
        sd.default.device[0] = self.input_devices_indices[
            self.input_devices.index(input_device)
        ]
        sd.default.device[1] = self.output_devices_indices[
            self.output_devices.index(output_device)
        ]
        printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
        printt("Output device: %s:%s", str(sd.default.device[1]), output_device)

    def get_device_samplerate(self):
        return int(
            sd.query_devices(device=sd.default.device[0])["default_samplerate"]
        )

    def get_device_channels(self):
        max_input_channels = sd.query_devices(device=sd.default.device[0])[
            "max_input_channels"
        ]
        max_output_channels = sd.query_devices(device=sd.default.device[1])[
            "max_output_channels"
        ]
        return min(max_input_channels, max_output_channels, 2)


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to the model checkpoint")
parser.add_argument("--config-path", type=str, default=None, help="Path to the vocoder checkpoint")
parser.add_argument("--fp16", type=str2bool, nargs="?", const=True, help="Whether to use fp16", default=True)
parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
args = parser.parse_args()
cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda" 
device = torch.device(cuda_target if torch.cuda.is_available() else "cpu")
gui = GUI(args)