# Tiny-ImageNet 64x64 Wide ResNet-34 + POT-QAT (Teacher–Student KD)
This repository provides a full training and quantization pipeline for deploying **Wide ResNet-34** on **Tiny-ImageNet (64×64)** with **Powers-of-Two Quantization (POT-QAT)** and **Knowledge Distillation (KD)**.

![DL_PROJ_CNN](https://github.com/user-attachments/assets/15296a73-4b73-476b-856c-b455aa1bd6a3)  ![DL_CNN_MODEL_DIAG](https://github.com/user-attachments/assets/cee973ed-8288-46ed-9d4e-fc2279cffed6)






The pipeline supports:
- Full-precision (FP32) teacher training  
- Quantization-Aware Training (QAT) with powers-of-two constraints  
- Teacher–student distillation  
- Checkpoint saving and weight export (`.pth`, `.npz`, `.txt`)  
- Visualization of training curves  
- Randomized evaluation on validation samples  

---

## 📂 Project Structure
tinyimagenet_wrn_pot/
│── train_pot_wrn.py # Full pipeline script (FP + POT-QAT)
│── tinyimagenet_wrn_pot_run/ # Output directory (checkpoints, plots, exports)
│ ├── best_model_fp.pth # Best FP teacher checkpoint
│ ├── final_model_pot.pth # Best POT student checkpoint
│ ├── weights_pot.npz # Exported POT weights
│ ├── weights_pot.txt # Human-readable POT weights
│ ├── model_meta.json # Metadata for quantization
│── README.md


##  Training Pipeline

1. Full-Precision Teacher Training
2. POT-QAT (Student with KD)

# Flags

TRAIN_FP          # Phase 1: Train full-precision teacher
TRAIN_POT         # Phase 2: POT QAT (student) with KD
TEST              # Run a small test at the end
SAVE_FP           # Save FP teacher checkpoint
SAVE_POT          # Save final POT student checkpoint
EXPORT_POT        # Export weights to .npz/.txt

📦 Dataset
We use Tiny-ImageNet (64×64) from HuggingFace Datasets:

👉 [Dataset Link (Tiny-ImageNet 64×64)](https://huggingface.co/datasets/zh-plus/tiny-imagenet/viewer/default/train?p=6)

💾 Pretrained Weights
Download pretrained checkpoints and exports here:

👉 [Trained Weights (Google Drive)](https://drive.google.com/drive/folders/1K92MTlDavK6B_w2fdO9Z_wknQAyGd7J1?usp=sharing)



📜 Citation
@misc{wrn_pot_qat_2025,
  title  = {Tiny-ImageNet 64x64 Wide ResNet-34 with POT-QAT},
  author = {Shay Saraf},
  year   = {2025},
}
