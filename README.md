# ARK.TASTE 🎨

**AI-Powered Interior Design Taste Profiling System**

ARK.TASTE is a deep learning system that analyzes user preferences in interior design and generates a personalized "User Taste DNA" based on 7 fundamental design elements. Built using the Multi-Modal Interior Style (MMIS) dataset.

## 📋 Overview

This implementation provides a complete pipeline for:
- **Phase 1**: CLIP-based feature extraction and style classification
- **Phase 2**: Multi-label style + 7 Element classification
- **User Taste DNA**: Personalized taste profile generation with confidence thresholds

### The 7 Design Elements

1. **Space Layout**: Open, Segmented, Cozy/Dense, Balanced
2. **Line Character**: Straight/Clean, Curved/Organic, Geometric/Strong, Mixed/Soft
3. **Form Type**: Rectilinear, Rounded/Soft, Sculptural/Statement, Functional/Minimal
4. **Light Profile**: Natural Dominant, Warm Ambient, Cool Bright, Contrasty Directional
5. **Color Palette**: Neutral Soft, Earth Tones, Bold Accented, Monochrome, Pastel Soft
6. **Texture Profile**: Smooth Minimal, Matte Natural, Glossy Polished, Rich Layered
7. **Pattern Intensity**: Minimal/None, Geometric Clean, Organic Soft, Ornate Detailed

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LY-ai-Lab/ARK.TASTE.git
cd ARK.TASTE

# Install dependencies
pip install -r requirements.txt
```

### Download MMIS Dataset

```bash
python download_mmis.py
```

This will download:
- 11 interior design styles × 5 room types = 55 categories
- Pre-computed VGG features
- JSON metadata files
- Dataset source: [MMIS Google Drive](https://drive.google.com/drive/folders/1FO_sNVZi757I_QBdwibPX--14O24ff0m)

### Generate Weak Labels

```bash
python weak_labeling.py
```

Generates pseudo-labels for the 7 Elements using rule-based heuristics from style annotations.

### Training

**Phase 2.1**: Train classification heads (freeze backbone)
```bash
python train.py --phase 1 --epochs 15 --batch_size 32 --lr 1e-3
```

**Phase 2.2**: Fine-tune top transformer blocks
```bash
python train.py --phase 2 --epochs 10 --batch_size 16 --lr 1e-5
```

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### Generate User Taste DNA

```bash
python taste_dna.py --user_id test_user --liked_images user_data/liked_images.json
```

Outputs a JSON with:
- Style distribution (multi-label)
- Element preferences across 7 dimensions
- Confidence score (stops at 85%+ or 20 images max)

## 📁 Project Structure

```
ARK.TASTE/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py              # Configuration settings
├── download_mmis.py       # MMIS dataset downloader
├── label_schema.py        # 7 Elements taxonomy
├── weak_labeling.py       # Pseudo-label generation
├── dataset.py             # PyTorch Dataset classes
├── ark_taste_model.py     # Model architecture
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── taste_dna.py          # User Taste DNA generator
└── checkpoints/           # Model checkpoints
```

## 🏗️ Architecture

### Backbone
- **CLIP ViT-B/16** (768-dim embeddings)
- Pre-trained on image-text pairs
- Frozen during Phase 2.1, top 4 blocks unfrozen in Phase 2.2

### Classification Heads
- **Style Head**: Multi-label BCE loss (11+ styles)
- **Element Heads**: 7 separate cross-entropy classifiers
- Total Loss: `BCE(styles) + Σ CE(elements)`

### Training Strategy
- **Phase 2.1**: Freeze backbone, train heads (10-20 epochs)
- **Phase 2.2**: Unfreeze top 4 ViT blocks, fine-tune (5-10 epochs)
- Optimizer: AdamW with weight decay
- Mixed precision training (AMP)

## 📊 MMIS Dataset

- **Source**: [Multi-Modal Interior Style Dataset](https://drive.google.com/drive/folders/1FO_sNVZi757I_QBdwibPX--14O24ff0m)
- **Styles**: Contemporary, Scandinavian, Industrial, Mid-Century Modern, Minimalist, Bohemian, Traditional, Rustic, Coastal, Eclectic, Art Deco
- **Rooms**: Living Room, Bedroom, Kitchen, Bathroom, Dining Room
- **Format**: Images + VGG features + JSON metadata

## 🎯 Usage Examples

### Training from Scratch

```python
import torch
from ark_taste_model import ARKTasteModel
from train import train_model

# Initialize model
model = ARKTasteModel(num_styles=11)

# Train Phase 2.1
train_model(model, phase=1, epochs=15, lr=1e-3)

# Train Phase 2.2
train_model(model, phase=2, epochs=10, lr=1e-5)
```

### Generate Taste DNA

```python
from taste_dna import generate_taste_dna

# User liked images
liked_images = [
    'path/to/image1.jpg',
    'path/to/image2.jpg',
    # ... more images
]

# Generate DNA
taste_dna = generate_taste_dna(
    model=model,
    images=liked_images,
    confidence_threshold=0.85
)

print(taste_dna)
# {
#   'styles': {'Contemporary': 0.78, 'Minimalist': 0.65, ...},
#   'elements': {
#     'space_layout': {'OPEN': 0.62, 'BALANCED': 0.31, ...},
#     'color_palette': {'NEUTRAL_SOFT': 0.71, ...},
#     ...
#   },
#   'confidence': 0.87,
#   'num_images': 12
# }
```

## 📈 Performance Metrics

- **Style Classification**: Multi-label accuracy, F1-score
- **Element Classification**: Per-element accuracy
- **User Taste DNA**: Confidence convergence (target: 85%)

## 🔗 Related Resources

- **Implementation Guide**: [Google Doc](https://docs.google.com/document/d/1p1L5oRRssGqQmc2kYsJHZE_92AoruOlOYhASO-AVeFo/edit)
- **Google AI Studio App**: [AI Studio](https://aistudio.google.com/apps/drive/1ZP_X4hacekUc80lm2NG4X5QCprmbUowL)
- **MMIS Dataset**: [Google Drive](https://drive.google.com/drive/folders/1FO_sNVZi757I_QBdwibPX--14O24ff0m)

## 📝 Citation

If you use this code or the MMIS dataset, please cite:

```bibtex
@misc{ark_taste_2026,
  title={ARK.TASTE: AI-Powered Interior Design Taste Profiling},
  author={LY-ai-Lab},
  year={2026},
  url={https://github.com/LY-ai-Lab/ARK.TASTE}
}
```

## 📄 License

This project is open source. See the repository for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

**Built with ❤️ by LY-ai-Lab**
