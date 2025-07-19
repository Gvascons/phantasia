import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    # Hardware Configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_vram_gb: float = 8.0  # RTX 4060 VRAM limit
    
    # Model Configurations
    whisper_model_size: str = "base"  # Options: tiny, base, small, medium
    whisper_compute_type: str = "float16"
    
    # Image Generation Settings
    image_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    image_resolution: tuple = (1024, 1024)
    image_inference_steps: int = 20
    image_guidance_scale: float = 7.5
    
    # Video Generation Settings
    video_model: str = "THUDM/CogVideoX-2b"  # Alternative: "alibaba-pai/Wan-2.1-1.3B"
    video_resolution: tuple = (480, 320)  # Lower res for RTX 4060
    video_frames: int = 16
    video_fps: int = 8
    
    # Audio Processing
    audio_chunk_duration: float = 5.0  # seconds
    audio_sample_rate: int = 16000
    vad_aggressiveness: int = 2  # Voice Activity Detection sensitivity
    
    # API Settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    max_concurrent_requests: int = 2
    
    # UI Settings
    ui_host: str = "127.0.0.1"
    ui_port: int = 7860
    
    # Storage
    models_dir: str = "./models"
    output_dir: str = "./outputs"
    temp_dir: str = "./temp"
    
    # Performance Optimizations
    use_fp8: bool = True  # Enable FP8 quantization for VRAM efficiency
    enable_attention_slicing: bool = True
    enable_memory_efficient_attention: bool = True
    enable_cpu_offload: bool = False
    
    # ComfyUI Integration
    comfyui_enabled: bool = True
    comfyui_workflows_dir: str = "./workflows/comfyui_workflows"
    
    class Config:
        env_file = ".env"
        env_prefix = "NVP_"  # Narration Visual Pipeline prefix

# Global settings instance
settings = Settings()

# Model configurations optimized for RTX 4060
MODEL_CONFIGS = {
    "image": {
        "memory_optimization": {
            "enable_sequential_cpu_offload": True,
            "enable_attention_slicing": "auto",
            "enable_memory_efficient_attention": True,
        },
        "generation_params": {
            "num_inference_steps": settings.image_inference_steps,
            "guidance_scale": settings.image_guidance_scale,
            "height": settings.image_resolution[1],
            "width": settings.image_resolution[0],
        }
    },
    "video": {
        "memory_optimization": {
            "enable_sequential_cpu_offload": True,
            "enable_model_cpu_offload": True,
        },
        "generation_params": {
            "num_frames": settings.video_frames,
            "height": settings.video_resolution[1],
            "width": settings.video_resolution[0],
            "fps": settings.video_fps,
        }
    },
    "audio": {
        "whisper_params": {
            "device": settings.device,
            "compute_type": settings.whisper_compute_type,
            "cpu_threads": 4,
            "num_workers": 1,
        }
    }
}

def get_available_vram():
    """Get available VRAM in GB"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0.0

def optimize_for_hardware():
    """Automatically optimize settings based on available hardware"""
    available_vram = get_available_vram()
    
    if available_vram < 6:
        # Ultra low VRAM mode
        settings.image_resolution = (512, 512)
        settings.video_resolution = (320, 240)
        settings.video_frames = 8
        settings.enable_cpu_offload = True
        settings.whisper_model_size = "tiny"
    elif available_vram < 8:
        # Low VRAM mode (RTX 4060)
        settings.image_resolution = (768, 768)
        settings.video_resolution = (480, 320)
        settings.video_frames = 16
        settings.whisper_model_size = "base"
    elif available_vram >= 12:
        # High VRAM mode
        settings.image_resolution = (1024, 1024)
        settings.video_resolution = (720, 480)
        settings.video_frames = 24
        settings.whisper_model_size = "small"
    
    return settings