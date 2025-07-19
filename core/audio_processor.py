import asyncio
import numpy as np
import sounddevice as sd
import threading
from typing import Optional, Callable, Generator
from faster_whisper import WhisperModel
import webrtcvad
from collections import deque
import time

from config.settings import settings, MODEL_CONFIGS

class AudioProcessor:
    """Real-time audio processing and transcription using faster-whisper"""
    
    def __init__(self, 
                 model_size: str = None,
                 device: str = None,
                 callback: Optional[Callable] = None):
        self.model_size = model_size or settings.whisper_model_size
        self.device = device or settings.device
        self.callback = callback
        
        # Initialize faster-whisper model
        whisper_params = MODEL_CONFIGS["audio"]["whisper_params"].copy()
        if self.device == "cpu":
            whisper_params.pop("compute_type", None)
        
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            **whisper_params
        )
        
        # Audio processing settings
        self.sample_rate = settings.audio_sample_rate
        self.chunk_duration = settings.audio_chunk_duration
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(settings.vad_aggressiveness)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.chunk_size * 3)  # 3x buffer
        self.is_recording = False
        self.transcription_thread = None
        
    def start_real_time_transcription(self) -> None:
        """Start real-time audio transcription"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.transcription_thread = threading.Thread(
            target=self._real_time_transcription_loop,
            daemon=True
        )
        self.transcription_thread.start()
        print("Started real-time transcription...")
    
    def stop_real_time_transcription(self) -> None:
        """Stop real-time audio transcription"""
        self.is_recording = False
        if self.transcription_thread:
            self.transcription_thread.join(timeout=2.0)
        print("Stopped real-time transcription")
    
    def _real_time_transcription_loop(self) -> None:
        """Main loop for real-time transcription"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            
            # Convert to int16 and add to buffer
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
            self.audio_buffer.extend(audio_int16)
        
        # Start audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=audio_callback,
            blocksize=1024
        ):
            while self.is_recording:
                if len(self.audio_buffer) >= self.chunk_size:
                    # Extract audio chunk
                    audio_chunk = np.array(list(self.audio_buffer)[:self.chunk_size])
                    
                    # Clear processed audio from buffer
                    for _ in range(self.chunk_size // 2):  # 50% overlap
                        if self.audio_buffer:
                            self.audio_buffer.popleft()
                    
                    # Check for voice activity
                    if self._has_voice_activity(audio_chunk):
                        # Transcribe chunk
                        transcription = self._transcribe_chunk(audio_chunk)
                        if transcription and self.callback:
                            self.callback(transcription)
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    def _has_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains voice activity"""
        try:
            # Convert to bytes for VAD
            audio_bytes = audio_chunk.astype(np.int16).tobytes()
            
            # Split into 30ms frames for VAD (requirement)
            frame_duration = 0.03  # 30ms
            frame_size = int(self.sample_rate * frame_duration)
            
            voice_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_chunk) - frame_size, frame_size):
                frame = audio_chunk[i:i + frame_size]
                frame_bytes = frame.astype(np.int16).tobytes()
                
                if len(frame_bytes) == frame_size * 2:  # 2 bytes per sample
                    if self.vad.is_speech(frame_bytes, self.sample_rate):
                        voice_frames += 1
                    total_frames += 1
            
            # Consider voice activity if >30% of frames contain speech
            return total_frames > 0 and (voice_frames / total_frames) > 0.3
            
        except Exception as e:
            print(f"VAD error: {e}")
            return True  # Default to processing if VAD fails
    
    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe audio chunk using faster-whisper"""
        try:
            # Convert to float32 and normalize
            audio_float = audio_chunk.astype(np.float32) / 32767.0
            
            # Transcribe
            segments, _ = self.model.transcribe(
                audio_float,
                language="en",  # Can be made configurable
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Combine all segments
            transcription = " ".join([segment.text.strip() for segment in segments])
            return transcription.strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def transcribe_file(self, audio_file_path: str) -> str:
        """Transcribe an audio file"""
        try:
            segments, info = self.model.transcribe(
                audio_file_path,
                language="en",
                vad_filter=True
            )
            
            transcription = " ".join([segment.text.strip() for segment in segments])
            
            print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            print(f"Duration: {info.duration:.2f}s")
            
            return transcription.strip()
            
        except Exception as e:
            print(f"File transcription error: {e}")
            return ""
    
    async def transcribe_audio_async(self, audio_data: np.ndarray) -> str:
        """Async transcription for API usage"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._transcribe_chunk, 
            audio_data
        )
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "chunk_duration": self.chunk_duration,
            "vad_aggressiveness": settings.vad_aggressiveness
        }

# Example usage and testing
if __name__ == "__main__":
    def transcription_callback(text: str):
        print(f"Transcribed: {text}")
    
    processor = AudioProcessor(callback=transcription_callback)
    
    try:
        processor.start_real_time_transcription()
        print("Speak into your microphone... (Press Ctrl+C to stop)")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        processor.stop_real_time_transcription()
        print("Real-time transcription stopped")