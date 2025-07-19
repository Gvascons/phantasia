# 🎨 Phantasia

**Transform your stories into stunning visuals with AI**

A production-ready, local AI pipeline that converts narrative text into high-quality images and videos using state-of-the-art models, optimized for RTX 4060 hardware.

![Demo](https://via.placeholder.com/800x400/667eea/ffffff?text=Phantasia)

## ✨ Features

- 🎭 **Intelligent Story Analysis** - Extracts characters, settings, mood, and visual elements
- 🖼️ **High-Quality Image Generation** - SDXL-powered image creation (768x768)
- 🎬 **Dynamic Video Generation** - Multi-frame interpolated videos from static scenes
- 💻 **RTX 4060 Optimized** - Memory-efficient processing under 8GB VRAM
- 🎨 **Professional UI** - Clean, modern web interface
- 🔒 **Fully Local** - No cloud dependencies, complete privacy

## 🚀 Quick Start

### Prerequisites
- NVIDIA RTX 4060 (or similar 8GB+ VRAM GPU)
- Python 3.11+
- CUDA 12.8+

### Installation & Launch

```bash
# 1. Clone and setup
git clone https://github.com/Gvascons/phantasia.git
cd phantasia

# 2. Create environment
python -m venv narration-env
source narration-env/bin/activate  # Linux/Mac
# OR: narration-env\Scripts\activate  # Windows

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch application
./run.sh
```

### Usage

1. **Open** `http://localhost:7860` in your browser
2. **Enter your story** in the text area
3. **Click "Analyze Story"** to extract visual elements
4. **Choose generation type** (auto/image/video)
5. **Click "Generate Visual"** to create content

## 🎯 How It Works

### Story Analysis
The pipeline intelligently analyzes your narrative text to extract:
- **Characters** (knights, wizards, dragons)
- **Settings** (castles, forests, mountains)
- **Objects** (swords, armor, magical items)
- **Mood** (dark, bright, mysterious)
- **Actions** (riding, casting, fighting)

### Visual Generation
Based on the analysis, the system:
- **For Static Scenes**: Generates high-quality SDXL images
- **For Dynamic Scenes**: Creates interpolated video sequences
- **Optimizes prompts** with style hints and quality modifiers

## 🛠️ Technical Stack

- **Image Generation**: Stable Diffusion XL with FP16 optimization
- **Video Generation**: Image-to-video interpolation pipeline
- **Text Processing**: Advanced NLP for visual element extraction
- **Interface**: Gradio with custom CSS styling
- **Backend**: FastAPI-ready architecture

## 📊 Performance

- **Image Generation**: ~15 seconds for 768x768 images
- **Video Generation**: ~15-20 minutes for 16-frame videos
- **VRAM Usage**: 6-7GB peak (RTX 4060 compatible)
- **Quality**: Production-ready visual output

## 🎨 Example Outputs

### Input Story
> "A brave knight in golden armor rides through an enchanted forest, his sword gleaming in the moonlight as magical creatures watch from the shadows."

### Generated Analysis
- **Scene Type**: Dynamic
- **Mood**: Mysterious
- **Elements**: Knight (character), Forest (setting), Sword (object)
- **Generation**: Video (16 frames)

## 🔧 Configuration

Key settings in `config/settings.py`:
- Image resolution (default: 768x768)
- Video frame count (default: 16)
- VRAM optimization levels
- Model selection

## 📁 Project Structure

```
phantasia/
├── 🎨 app.py                    # Production application
├── 🚀 run.sh                    # Launch script
├── 🧠 core/                     # Core processing modules
│   ├── text_extractor.py        # Story analysis
│   ├── image_generator.py       # SDXL integration
│   └── video_generator.py       # Video creation
├── ⚙️ config/                   # Configuration
│   └── settings.py              # System settings
├── 📦 requirements.txt          # Dependencies
└── 📖 README.md                 # This file
```

## 🎭 Supported Story Types

- **Fantasy**: Knights, wizards, dragons, magic
- **Adventure**: Journeys, quests, exploration
- **Nature**: Landscapes, forests, mountains
- **Architecture**: Castles, towers, villages
- **Characters**: Heroes, villains, mystical beings

## 🔍 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce image resolution in config/settings.py
image_resolution = (512, 512)
```

**Slow Generation**
- First run downloads models (~6GB)
- Subsequent runs are much faster
- Video generation inherently takes longer

**Model Loading Errors**
```bash
# Clear cache and restart
rm -rf ~/.cache/huggingface/
python app.py
```

## 📈 Future Enhancements

- [ ] Real-time audio narration input
- [ ] Multiple art styles (anime, realistic, artistic)
- [ ] Batch processing capabilities
- [ ] Advanced video effects
- [ ] Custom model fine-tuning

## 🤝 Contributing

This project uses production-ready, modular architecture. Contributions welcome!

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- [Stability AI](https://stability.ai/) for Stable Diffusion XL
- [Hugging Face](https://huggingface.co/) for model hosting
- [Gradio](https://gradio.app/) for the UI framework

---

**Transform your imagination into visual reality with AI! 🚀**