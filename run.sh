#!/bin/bash

echo "🎨 PHANTASIA - Production Launch"
echo "================================="

# Activate environment
echo "📦 Activating environment..."
source narration-env/bin/activate

# Check system
echo "🔍 System check..."
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"

echo ""
echo "🚀 Launching Phantasia..."
echo "📍 Access at: http://localhost:7860"
echo ""

# Launch production app
python app.py