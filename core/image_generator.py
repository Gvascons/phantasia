import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    DPMSolverMultistepScheduler,
    AutoencoderKL
)
from typing import List, Optional, Dict, Any, Union
import numpy as np
from PIL import Image
import gc
import logging
from pathlib import Path

from config.settings import settings, MODEL_CONFIGS
from core.text_extractor import SceneDescription

class ImageGenerator:
    """SDXL-based image generation optimized for RTX 4060"""
    
    def __init__(self, 
                 model_name: str = None,
                 device: str = None,
                 enable_optimizations: bool = True):
        
        self.model_name = model_name or settings.image_model
        self.device = device or settings.device
        self.enable_optimizations = enable_optimizations
        
        self.pipeline = None
        self.is_loaded = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Memory optimization settings
        self.memory_config = MODEL_CONFIGS["image"]["memory_optimization"]
        self.generation_config = MODEL_CONFIGS["image"]["generation_params"]
        
    def load_model(self, force_reload: bool = False) -> None:
        """Load SDXL model with RTX 4060 optimizations"""
        
        if self.is_loaded and not force_reload:
            return
            
        self.logger.info(f"Loading SDXL model: {self.model_name}")
        
        try:
            # Free memory first
            self._free_memory()
            
            # Load VAE separately for better memory management
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sdxl-vae",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
            
            # Load main pipeline
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                vae=vae,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Apply memory optimizations
            if self.enable_optimizations:
                self._apply_optimizations()
            
            # Set scheduler for better quality/speed tradeoff
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            self.is_loaded = True
            self.logger.info("SDXL model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load SDXL model: {e}")
            raise
    
    def _apply_optimizations(self) -> None:
        """Apply RTX 4060-specific optimizations"""
        
        # Enable attention slicing for lower VRAM usage
        if self.memory_config.get("enable_attention_slicing"):
            self.pipeline.enable_attention_slicing("auto")
            self.logger.info("Enabled attention slicing")
        
        # Enable memory efficient attention
        if self.memory_config.get("enable_memory_efficient_attention"):
            self.pipeline.enable_xformers_memory_efficient_attention()
            self.logger.info("Enabled memory efficient attention")
        
        # Enable CPU offloading if needed - simplified for compatibility
        if self.memory_config.get("enable_sequential_cpu_offload"):
            try:
                self.pipeline.enable_model_cpu_offload()
                self.logger.info("Enabled model CPU offload")
            except Exception as e:
                self.logger.warning(f"CPU offload failed, continuing without: {e}")
        
        # Enable VAE slicing for large images
        self.pipeline.enable_vae_slicing()
        
        # Compile model for faster inference (PyTorch 2.0+) - disabled due to compatibility issues
        # if hasattr(torch, 'compile') and self.device == "cuda":
        #     try:
        #         self.pipeline.unet = torch.compile(self.pipeline.unet)
        #         self.logger.info("Compiled UNet for faster inference")
        #     except Exception as e:
        #         self.logger.warning(f"Could not compile model: {e}")
        self.logger.info("Model compilation disabled for compatibility")
    
    def generate_image(self, 
                      scene_description: SceneDescription,
                      num_images: int = 1,
                      seed: Optional[int] = None,
                      custom_params: Optional[Dict[str, Any]] = None) -> List[Image.Image]:
        """Generate images from scene description"""
        
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
        
        self.logger.info(f"Generating {num_images} image(s) with prompt: {prompt[:100]}...")
        
        try:
            # Generate images
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=scene_description.negative_prompt,
                    num_images_per_prompt=num_images,
                    generator=generator,
                    **params
                )
            
            images = result.images
            self.logger.info(f"Successfully generated {len(images)} image(s)")
            
            return images
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error("CUDA out of memory. Trying with reduced settings...")
            return self._generate_with_reduced_settings(scene_description, num_images, seed)
        
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise
    
    def _generate_with_reduced_settings(self, 
                                      scene_description: SceneDescription,
                                      num_images: int,
                                      seed: Optional[int]) -> List[Image.Image]:
        """Fallback generation with reduced memory usage"""
        
        self.logger.info("Attempting generation with reduced settings...")
        
        # Reduce image size
        reduced_params = self.generation_config.copy()
        reduced_params['height'] = 512
        reduced_params['width'] = 512
        reduced_params['num_inference_steps'] = 15
        
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
                    num_images_per_prompt=num_images,
                    generator=generator,
                    **reduced_params
                )
            
            return result.images
            
        except Exception as e:
            self.logger.error(f"Reduced settings generation also failed: {e}")
            raise
    
    def _build_complete_prompt(self, scene_description: SceneDescription) -> str:
        """Build complete prompt from scene description"""
        
        prompt_parts = [scene_description.main_prompt]
        
        # Add style hints
        if scene_description.style_hints:
            prompt_parts.extend(scene_description.style_hints)
        
        # Add quality modifiers
        quality_modifiers = [
            "masterpiece",
            "best quality", 
            "highly detailed",
            "sharp focus",
            "professional"
        ]
        prompt_parts.extend(quality_modifiers)
        
        return ", ".join(prompt_parts)
    
    def generate_variations(self, 
                           base_image: Image.Image,
                           scene_description: SceneDescription,
                           strength: float = 0.7,
                           num_variations: int = 3) -> List[Image.Image]:
        """Generate variations of an existing image"""
        
        # This would require img2img pipeline
        # Implementation would depend on specific requirements
        raise NotImplementedError("Image variations not yet implemented")
    
    def upscale_image(self, image: Image.Image, scale_factor: int = 2) -> Image.Image:
        """Upscale image using simple interpolation (placeholder for real upscaler)"""
        
        new_size = (image.width * scale_factor, image.height * scale_factor)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    
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
            self.logger.info("Model unloaded")
    
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
    
    def batch_generate(self, 
                      scene_descriptions: List[SceneDescription],
                      images_per_scene: int = 1,
                      save_path: Optional[Path] = None) -> List[List[Image.Image]]:
        """Generate multiple images in batch"""
        
        all_images = []
        
        for i, scene_desc in enumerate(scene_descriptions):
            self.logger.info(f"Processing scene {i+1}/{len(scene_descriptions)}")
            
            try:
                images = self.generate_image(scene_desc, num_images=images_per_scene)
                all_images.append(images)
                
                # Save if path provided
                if save_path:
                    for j, img in enumerate(images):
                        img_path = save_path / f"scene_{i+1}_image_{j+1}.png"
                        img.save(img_path)
                
            except Exception as e:
                self.logger.error(f"Failed to generate scene {i+1}: {e}")
                all_images.append([])  # Empty list for failed generation
        
        return all_images

# Example usage and testing
if __name__ == "__main__":
    from core.text_extractor import TextExtractor
    
    # Initialize components
    extractor = TextExtractor()
    generator = ImageGenerator()
    
    # Test text
    test_text = "A majestic castle on a hilltop under a starry night sky, with a lone knight approaching on horseback"
    
    # Extract scene description
    scene_desc = extractor.extract_scene_description(test_text)
    
    print(f"Scene description: {scene_desc.main_prompt}")
    print(f"Generation type: {scene_desc.generation_type}")
    
    # Generate image
    try:
        images = generator.generate_image(scene_desc, num_images=1, seed=42)
        print(f"Generated {len(images)} image(s)")
        
        # Save first image
        if images:
            images[0].save("test_generation.png")
            print("Saved test image as 'test_generation.png'")
            
        # Print memory usage
        memory_info = generator.get_memory_usage()
        print(f"Memory usage: {memory_info}")
        
    except Exception as e:
        print(f"Generation failed: {e}")
    
    finally:
        generator.unload_model()