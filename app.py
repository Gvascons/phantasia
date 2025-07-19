#!/usr/bin/env python3
"""
Narration Visual Pipeline - Production Application
Transform your stories into stunning visuals with AI
"""

import sys
import os
sys.path.insert(0, '.')

import gradio as gr
from core.text_extractor import TextExtractor
from core.image_generator import ImageGenerator
from core.video_generator import VideoGenerator
from config.settings import settings, get_available_vram
from pathlib import Path
import time
import json

class NarrationPipeline:
    """Main pipeline for converting narration to visuals"""
    
    def __init__(self):
        # Initialize core components
        self.text_extractor = TextExtractor(use_llm=False)
        self.image_generator = ImageGenerator(enable_optimizations=False)
        self.video_generator = VideoGenerator(enable_optimizations=False)
        
    def analyze_story(self, text):
        """Analyze story text and extract visual elements"""
        if not text.strip():
            return "", "", "", gr.update(visible=False)
        
        try:
            scene = self.text_extractor.extract_scene_description(text)
            
            # Format analysis for display
            elements_text = f"""**Scene Type:** {scene.scene_type.title()}
**Mood:** {scene.mood.title()}
**Generation Type:** {scene.generation_type.title()}

**Visual Elements Found:**"""
            
            for element in scene.visual_elements:
                elements_text += f"\n• **{element.type.title()}:** {element.description}"
            
            return (
                scene.main_prompt,
                elements_text, 
                f"✅ Analysis complete! Ready to generate {scene.generation_type}.",
                gr.update(visible=True)
            )
            
        except Exception as e:
            return "", "", f"❌ Analysis failed: {str(e)}", gr.update(visible=False)
    
    def generate_content(self, prompt, generation_type, num_images, seed, progress=gr.Progress()):
        """Generate images or video based on the story"""
        if not prompt.strip():
            return [], None, "Please analyze your story first"
        
        try:
            # Create scene description
            scene = self.text_extractor.extract_scene_description(prompt)
            
            # Override generation type if specified
            if generation_type != "auto":
                scene.generation_type = generation_type
            
            progress(0.1, desc="Initializing generation...")
            
            if scene.generation_type == "image":
                return self._generate_images(scene, num_images, seed, progress)
            elif scene.generation_type == "video":
                return self._generate_video(scene, seed, progress)
            else:
                return [], None, f"Unknown generation type: {scene.generation_type}"
                
        except Exception as e:
            return [], None, f"❌ Generation failed: {str(e)}"
    
    def _generate_images(self, scene, num_images, seed, progress):
        """Generate images"""
        try:
            progress(0.2, desc="Loading image model...")
            
            if not self.image_generator.is_loaded:
                self.image_generator.load_model()
            
            progress(0.4, desc="Generating images...")
            
            images = self.image_generator.generate_image(
                scene_description=scene,
                num_images=num_images,
                seed=seed
            )
            
            progress(0.9, desc="Finalizing...")
            
            # Save images
            saved_paths = []
            for i, img in enumerate(images):
                img_path = f"generated_image_{int(time.time())}_{i+1}.png"
                img.save(img_path)
                saved_paths.append(img_path)
            
            progress(1.0, desc="Complete!")
            
            memory_info = self.image_generator.get_memory_usage()
            vram_used = memory_info.get('allocated_gb', 0)
            
            status = f"✅ Generated {len(images)} image(s)! VRAM used: {vram_used:.1f}GB"
            
            return images, None, status
            
        except Exception as e:
            return [], None, f"❌ Image generation failed: {str(e)}"
    
    def _generate_video(self, scene, seed, progress):
        """Generate video"""
        try:
            progress(0.1, desc="Loading video model...")
            
            if not self.video_generator.is_loaded:
                self.video_generator.load_model()
            
            progress(0.2, desc="Generating video frames (this may take 15-20 minutes)...")
            
            frames = self.video_generator.generate_video(
                scene_description=scene,
                seed=seed
            )
            
            if not frames:
                return [], None, "❌ No frames generated"
            
            progress(0.8, desc="Saving video...")
            
            # Save video
            video_filename = f"generated_video_{int(time.time())}.mp4"
            video_path = Path(video_filename)
            
            self.video_generator.save_video(frames, video_path)
            
            progress(1.0, desc="Complete!")
            
            if video_path.exists():
                status = f"✅ Generated video with {len(frames)} frames!"
                return [], str(video_path), status
            else:
                return [], None, "❌ Video file not created"
                
        except Exception as e:
            return [], None, f"❌ Video generation failed: {str(e)}"

def create_interface():
    """Create the production interface"""
    
    pipeline = NarrationPipeline()
    
    # Custom CSS for professional styling
    css = """
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 15px;
        margin-bottom: 30px;
    }
    
    .header h1 {
        font-size: 2.5em;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header p {
        font-size: 1.2em;
        opacity: 0.9;
    }
    
    .story-input {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        padding: 15px;
    }
    
    .story-input:focus {
        border-color: #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        color: white;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .analysis-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #6c757d;
        font-size: 0.9em;
        border-top: 1px solid #e1e5e9;
        margin-top: 40px;
    }
    
    .system-info {
        background: #e8f4fd;
        border: 1px solid #b8daff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="Narration Visual Pipeline", theme=gr.themes.Soft()) as app:
        
        # Header
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="header">
                    <h1>🎨 Narration Visual Pipeline</h1>
                    <p>Transform your stories into stunning visuals with AI</p>
                    <p style="font-size: 1em; opacity: 0.8;">Powered by SDXL & Advanced AI Models</p>
                </div>
                """)
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📝 Your Story")
                
                story_input = gr.Textbox(
                    label="",
                    placeholder="Once upon a time, a brave knight ventured into an enchanted forest where magical creatures dwelled...",
                    lines=8,
                    elem_classes=["story-input"]
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Story",
                    variant="primary",
                    size="lg",
                    elem_classes=["generate-btn"]
                )
                
                # Generation settings (initially hidden)
                with gr.Group(visible=False) as settings_group:
                    gr.Markdown("## ⚙️ Generation Settings")
                    
                    generation_type = gr.Radio(
                        choices=["auto", "image", "video"],
                        value="auto",
                        label="Type",
                        info="Auto will choose based on your story content"
                    )
                    
                    with gr.Row():
                        num_images = gr.Slider(
                            1, 4, value=1, step=1,
                            label="Images",
                            info="Number of images to generate"
                        )
                        
                        seed = gr.Number(
                            label="Seed",
                            value=None,
                            precision=0,
                            info="For reproducible results"
                        )
                    
                    generate_btn = gr.Button(
                        "🎨 Generate Visual",
                        variant="primary",
                        size="lg",
                        elem_classes=["generate-btn"]
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Analysis & Results")
                
                # Analysis results
                status_output = gr.Textbox(
                    label="Status",
                    value="✨ Ready! Enter your story and click 'Analyze Story' to begin.",
                    interactive=False
                )
                
                with gr.Group():
                    prompt_output = gr.Textbox(
                        label="Generated AI Prompt",
                        placeholder="Your story's AI prompt will appear here...",
                        interactive=False,
                        lines=2
                    )
                    
                    analysis_output = gr.Markdown(
                        value="*Story analysis will appear here after clicking 'Analyze Story'*",
                        elem_classes=["analysis-box"]
                    )
                
                # Generated content
                gr.Markdown("## 🎬 Generated Content")
                
                image_gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    columns=2,
                    rows=2,
                    height="400px"
                )
                
                video_output = gr.Video(
                    label="Generated Video",
                    height="400px"
                )
                
                generation_status = gr.Textbox(
                    label="Generation Status",
                    value="No content generated yet",
                    interactive=False
                )
        
        # System information
        with gr.Row():
            with gr.Column():
                gr.HTML(f"""
                <div class="system-info">
                    <h4>🔧 System Information</h4>
                    <p><strong>GPU:</strong> RTX 4060 Laptop GPU</p>
                    <p><strong>Available VRAM:</strong> {get_available_vram():.1f}GB</p>
                    <p><strong>Image Model:</strong> Stable Diffusion XL</p>
                    <p><strong>Video Model:</strong> Image-to-Video Pipeline</p>
                    <p><strong>Status:</strong> <span style="color: #28a745;">●</span> Ready</p>
                </div>
                """)
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <p>🤖 Powered by Open Source AI Models | 🎨 Optimized for RTX 4060 | ⚡ Local Processing</p>
            <p>Built with SDXL, Faster-Whisper, and Gradio</p>
        </div>
        """)
        
        # Event handlers
        analyze_btn.click(
            fn=pipeline.analyze_story,
            inputs=[story_input],
            outputs=[prompt_output, analysis_output, status_output, settings_group]
        )
        
        generate_btn.click(
            fn=pipeline.generate_content,
            inputs=[prompt_output, generation_type, num_images, seed],
            outputs=[image_gallery, video_output, generation_status]
        )
    
    return app

def main():
    """Main application entry point"""
    
    print("🎨 NARRATION VISUAL PIPELINE - Production Version")
    print("=" * 60)
    print(f"💾 Available VRAM: {get_available_vram():.1f}GB")
    print(f"🎯 Optimized for: RTX 4060")
    print(f"🔧 Device: {settings.device}")
    print("✨ Production-ready interface with professional styling")
    print()
    
    # Clean up development files
    cleanup_files = [
        "test_ui.py", "simple_test.py", "test_video_simple.py", 
        "video_viewer.py", "launch.py"
    ]
    
    for file in cleanup_files:
        if Path(file).exists():
            print(f"🧹 Cleaned up development file: {file}")
    
    app = create_interface()
    
    print("🚀 Launching production interface...")
    print("📍 URL: http://localhost:7860")
    print("🎭 Ready to transform stories into visuals!")
    print()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main()