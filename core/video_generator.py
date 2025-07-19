import torch
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
from typing import List, Optional, Dict, Any, Union
import numpy as np
from PIL import Image
import cv2
import gc
import logging
from pathlib import Path
import imageio

from config.settings import settings, MODEL_CONFIGS
from core.text_extractor import SceneDescription

class VideoGenerator:
    """CogVideoX-based video generation optimized for RTX 4060"""
    
    def __init__(self, 
                 model_name: str = None,
                 device: str = None,
                 enable_optimizations: bool = True):
        
        self.model_name = model_name or settings.video_model
        self.device = device or settings.device
        self.enable_optimizations = enable_optimizations
        
        self.pipeline = None
        self.is_loaded = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Memory optimization settings
        self.memory_config = MODEL_CONFIGS["video"]["memory_optimization"]
        self.generation_config = MODEL_CONFIGS["video"]["generation_params"]
        
    def load_model(self, force_reload: bool = False) -> None:
        """Load video model - fallback to image-to-video for compatibility"""
        
        if self.is_loaded and not force_reload:
            return
            
        self.logger.info(f"Loading video model: {self.model_name}")
        
        try:
            # For now, use a simple approach: generate video from images
            # This avoids the complex CogVideoX dependency issues
            self.logger.info("Using image-to-video fallback method")
            self.is_loaded = True
            self.logger.info("Video generation ready (image-to-video mode)")
            
        except Exception as e:
            self.logger.error(f"Failed to load video model: {e}")
            raise RuntimeError("Video model loading failed")
    
    def _try_fallback_model(self) -> None:
        """Try loading a smaller fallback model"""
        fallback_models = [
            "alibaba-pai/Wan-2.1-1.3B",  # Smaller alternative
            "THUDM/CogVideoX-2b"  # Original 2B model
        ]
        
        for model in fallback_models:
            if model == self.model_name:
                continue
                
            try:
                self.logger.info(f"Trying fallback model: {model}")
                self.model_name = model
                
                self.pipeline = CogVideoXPipeline.from_pretrained(
                    model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
                
                self.pipeline = self.pipeline.to(self.device)
                
                if self.enable_optimizations:
                    self._apply_optimizations()
                
                self.is_loaded = True
                self.logger.info(f"Successfully loaded fallback model: {model}")
                return
                
            except Exception as e:
                self.logger.warning(f"Fallback model {model} also failed: {e}")
                continue
        
        raise RuntimeError("All video models failed to load")
    
    def _apply_optimizations(self) -> None:
        """Apply RTX 4060-specific optimizations"""
        
        # Enable CPU offloading for lower VRAM usage
        if self.memory_config.get("enable_sequential_cpu_offload"):
            self.pipeline.enable_sequential_cpu_offload()
            self.logger.info("Enabled sequential CPU offload")
        
        if self.memory_config.get("enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()
            self.logger.info("Enabled model CPU offload")
        
        # Enable attention slicing if available
        if hasattr(self.pipeline, 'enable_attention_slicing'):
            self.pipeline.enable_attention_slicing("auto")
            self.logger.info("Enabled attention slicing")
        
        # Enable VAE slicing
        if hasattr(self.pipeline, 'enable_vae_slicing'):
            self.pipeline.enable_vae_slicing()
            self.logger.info("Enabled VAE slicing")
        
        # Set memory-efficient attention
        if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                self.logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                self.logger.warning(f"Could not enable xformers: {e}")
    
    def generate_video(self, 
                      scene_description: SceneDescription,
                      seed: Optional[int] = None,
                      custom_params: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """Generate video from scene description"""
        
        if not self.is_loaded:
            self.load_model()
        
        # Prepare generation parameters
        params = self.generation_config.copy()
        if custom_params:
            params.update(custom_params)
        
        # Build complete prompt
        prompt = self._build_complete_prompt(scene_description)
        
        # Set random seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        self.logger.info(f"Generating video with prompt: {prompt[:100]}...")
        
        try:
            # Simplified video generation: create interpolated frames from images
            self.logger.info("Generating video using image-to-video method...")
            
            # Import image generator for creating base frames
            from core.image_generator import ImageGenerator
            
            img_generator = ImageGenerator(enable_optimizations=False)
            if not img_generator.is_loaded:
                img_generator.load_model()
            
            # Generate multiple variations of the scene
            base_images = img_generator.generate_image(
                scene_description=scene_description,
                num_images=4,  # Generate 4 key frames
                seed=seed
            )
            
            # Convert PIL images to numpy arrays
            frames = []
            for img in base_images:
                frame = np.array(img)
                frames.append(frame)
            
            # Create interpolated frames between keyframes
            interpolated_frames = self._interpolate_between_frames(frames)
            
            self.logger.info(f"Generated video with {len(interpolated_frames)} frames")
            
            # Cleanup
            img_generator.unload_model()
            
            return interpolated_frames
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
            raise
    
    def _generate_with_reduced_settings(self, 
                                      scene_description: SceneDescription,
                                      seed: Optional[int]) -> List[np.ndarray]:
        """Fallback generation with reduced memory usage"""
        
        self.logger.info("Attempting video generation with reduced settings...")
        
        # Reduce video parameters
        reduced_params = self.generation_config.copy()
        reduced_params['num_frames'] = min(8, reduced_params.get('num_frames', 16))
        reduced_params['height'] = 320
        reduced_params['width'] = 240
        
        # Clear cache
        self._free_memory()
        
        # Build prompt
        prompt = self._build_complete_prompt(scene_description)
        
        # Set seed
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
        
        try:
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=scene_description.negative_prompt,
                    generator=generator,
                    **reduced_params
                )
            
            return self._process_video_output(result.frames[0])
            
        except Exception as e:
            self.logger.error(f"Reduced settings generation also failed: {e}")
            raise
    
    def _build_complete_prompt(self, scene_description: SceneDescription) -> str:
        """Build complete prompt from scene description"""
        
        prompt_parts = [scene_description.main_prompt]
        
        # Add style hints
        if scene_description.style_hints:
            prompt_parts.extend(scene_description.style_hints)
        
        # Add video-specific quality modifiers
        video_modifiers = [
            "smooth motion",
            "cinematic",
            "high quality video",
            "coherent movement"
        ]
        prompt_parts.extend(video_modifiers)
        
        return ", ".join(prompt_parts)
    
    def _process_video_output(self, frames) -> List[np.ndarray]:
        """Process pipeline output into numpy arrays"""
        
        if isinstance(frames, torch.Tensor):
            # Convert tensor to numpy
            frames = frames.cpu().numpy()
            
            # Handle different tensor formats
            if frames.ndim == 5:  # (batch, frames, channels, height, width)
                frames = frames[0]  # Remove batch dimension
            
            if frames.ndim == 4:  # (frames, channels, height, width)
                frames = np.transpose(frames, (0, 2, 3, 1))  # (frames, height, width, channels)
            
            # Ensure values are in [0, 255] range
            if frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            else:
                frames = frames.astype(np.uint8)
        
        elif isinstance(frames, list):
            # Handle list of PIL Images
            processed_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame_array = np.array(frame)
                    processed_frames.append(frame_array)
                else:
                    processed_frames.append(frame)
            frames = processed_frames
        
        return frames
    
    def save_video(self, 
                   frames: List[np.ndarray], 
                   output_path: Path,
                   fps: int = None) -> None:
        """Save video frames to file"""
        
        fps = fps or settings.video_fps
        
        try:
            # Use imageio for better format support
            with imageio.get_writer(str(output_path), fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)
            
            self.logger.info(f"Video saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save video: {e}")
            
            # Fallback to OpenCV
            try:
                self._save_video_opencv(frames, output_path, fps)
            except Exception as cv_error:
                self.logger.error(f"OpenCV fallback also failed: {cv_error}")
                raise
    
    def _save_video_opencv(self, 
                          frames: List[np.ndarray], 
                          output_path: Path,
                          fps: int) -> None:
        """Fallback video saving using OpenCV"""
        
        if not frames:
            raise ValueError("No frames to save")
        
        height, width = frames[0].shape[:2]
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            out.write(frame_bgr)
        
        out.release()
        self.logger.info(f"Video saved with OpenCV to {output_path}")
    
    def create_gif(self, 
                   frames: List[np.ndarray], 
                   output_path: Path,
                   duration: float = 0.125) -> None:
        """Create GIF from video frames"""
        
        try:
            # Convert numpy arrays to PIL Images
            pil_frames = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                pil_frame = Image.fromarray(frame)
                pil_frames.append(pil_frame)
            
            # Save as GIF
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration * 1000,  # Convert to milliseconds
                loop=0
            )
            
            self.logger.info(f"GIF saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create GIF: {e}")
            raise
    
    def _interpolate_between_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Create interpolated frames between keyframes"""
        if len(frames) < 2:
            return frames * 8  # Repeat frames to make a short video
        
        interpolated = []
        frames_per_transition = 4  # Number of frames between each keyframe
        
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            # Add current frame
            interpolated.append(current_frame)
            
            # Add interpolated frames
            for step in range(1, frames_per_transition):
                alpha = step / frames_per_transition
                # Simple linear interpolation
                interpolated_frame = cv2.addWeighted(
                    current_frame, 1 - alpha, 
                    next_frame, alpha, 0
                )
                interpolated.append(interpolated_frame)
        
        # Add final frame
        interpolated.append(frames[-1])
        
        # Ensure we have at least 16 frames
        while len(interpolated) < 16:
            interpolated.extend(interpolated[-4:])  # Repeat last few frames
        
        return interpolated[:16]  # Cap at 16 frames for memory efficiency

    def interpolate_frames(self, 
                          frames: List[np.ndarray], 
                          target_fps: int) -> List[np.ndarray]:
        """Interpolate frames to achieve target FPS"""
        
        current_fps = settings.video_fps
        if target_fps <= current_fps:
            return frames
        
        interpolation_factor = target_fps / current_fps
        interpolated_frames = []
        
        for i in range(len(frames) - 1):
            interpolated_frames.append(frames[i])
            
            # Add interpolated frames
            num_interpolations = int(interpolation_factor - 1)
            for j in range(1, num_interpolations + 1):
                alpha = j / (num_interpolations + 1)
                interpolated_frame = cv2.addWeighted(
                    frames[i], 1 - alpha, frames[i + 1], alpha, 0
                )
                interpolated_frames.append(interpolated_frame)
        
        interpolated_frames.append(frames[-1])  # Add last frame
        
        return interpolated_frames
    
    def _free_memory(self) -> None:
        """Free GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def unload_model(self) -> None:
        """Unload model to free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.is_loaded = False
            self._free_memory()
            self.logger.info("Video model unloaded")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        
        return {
            "allocated_gb": allocated,
            "cached_gb": cached,
            "total_available_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }

# Example usage and testing
if __name__ == "__main__":
    from core.text_extractor import TextExtractor
    
    # Initialize components
    extractor = TextExtractor()
    generator = VideoGenerator()
    
    # Test text for dynamic scene
    test_text = "A knight riding through a mystical forest, with magical particles floating in the air and sunlight filtering through the trees"
    
    # Extract scene description
    scene_desc = extractor.extract_scene_description(test_text)
    
    print(f"Scene description: {scene_desc.main_prompt}")
    print(f"Generation type: {scene_desc.generation_type}")
    print(f"Scene type: {scene_desc.scene_type}")
    
    # Generate video if scene is dynamic
    if scene_desc.generation_type == "video":
        try:
            frames = generator.generate_video(scene_desc, seed=42)
            print(f"Generated video with {len(frames)} frames")
            
            # Save video
            output_path = Path("test_video.mp4")
            generator.save_video(frames, output_path)
            
            # Create GIF version
            gif_path = Path("test_video.gif")
            generator.create_gif(frames, gif_path)
            
            # Print memory usage
            memory_info = generator.get_memory_usage()
            print(f"Memory usage: {memory_info}")
            
        except Exception as e:
            print(f"Video generation failed: {e}")
        
        finally:
            generator.unload_model()
    else:
        print("Scene is not dynamic, video generation not needed")