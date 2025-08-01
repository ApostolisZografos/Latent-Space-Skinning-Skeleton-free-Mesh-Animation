# Latent-Space Skinning: Skeleton-free Mesh Animation

This repository contains the code and deep learning methods developed for my diploma thesis on **motion synthesis** from animation data.

The core idea is to train deep learning models (using PyTorch) on different motion styles (e.g., walking, jumping), and then synthesize new animations by blending these learned styles into new transitions or hybrid motions.

## 🔧 Technologies Used

- Python
- PyTorch
- CUDA
- NumPy
- Matplotlib

## 📁 Structure

The main logic is in the `main_controller.py` file, which provides a command-line interface with the following options:

- `train`: Train a model on a specific motion category
- `synthesis`: Synthesize motion between categories (e.g., blend walking and jumping)
- `test`: Test the model evaluation
- `category`: Category moodel trainning
- `Evaluate categories`: Make Synthesis for categorized model



## Warning
- in order to run the code you need to initialize these libraries (better make a Conda Enviroment)
  
![image](https://github.com/user-attachments/assets/e309653e-7037-460c-9143-a291f951dda0)


