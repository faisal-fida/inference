class VoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.port.onmessage = (event) => {
      // Handle messages from the main thread if necessary
    };
  }

  process(inputs, outputs, parameters) {
    const inputChannelData = inputs[0][0]; // First input, first channel
    if (inputChannelData) {
      // Convert Float32Array to ArrayBuffer
      const inputBuffer = inputChannelData.buffer.slice(0);
      this.port.postMessage(inputBuffer);
    }
    return true;
  }
}

registerProcessor('voice-processor', VoiceProcessor);