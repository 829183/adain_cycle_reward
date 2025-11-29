# AdaIN Cycle Consistency Reward

Quantifying content preservation in AdaIN style transfer using invertible statistics and perceptual similarity.

<!-- ![teaser](assets/teaser.jpg) -->

## 🎯 Goal
Evaluate how well AdaIN preserves content under different:
- Style images
- Content images
- Style strength (α)

Using **cycle consistency reward**:  
`reward = 1 - DREAMSim(original, reconstructed)`

## 🛠️ Setup
```bash
git clone https://github.com/your-username/adain_cycle_reward.git
cd adain_cycle_reward
conda env create -f environment.yml
conda activate acrwd
```

## 📚 References
- Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. ICCV.
- Bahng, H., Chan, C., Durand, F., & Isola, P. (2025). Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences. ICCV.
- DREAMSim: https://github.com/ssundaram21/dreamsim