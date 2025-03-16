import os
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from safetensors.torch import load_file

# Read model paths from environment variables
# model_paths = json.loads(os.getenv('MODEL_PATHS', '{}'))


def load_lora_params(ckpt_path):
    ckpt = torch.load(ckpt_path,map_location='cpu')
    # lora_B0 = [param.data.numpy().flatten().reshape(-1) for key, param in ckpt.items() if 'lora_B0' in key]
    # lora_B1 = [param.data.numpy().flatten().reshape(-1) for key, param in ckpt.items() if 'lora_B1' in key]
    # lora_B2 = [param.data.numpy().flatten().reshape(-1) for key, param in ckpt.items() if 'lora_B2' in key]
    lora_B0 = []
    lora_B1 = []
    lora_B2 = []
    for name, param in ckpt.items():
        if 'lora_B0' in name:
            lora_B0 += param.flatten().tolist()
        elif 'lora_B1' in name:
            lora_B1 += param.flatten().tolist()
        elif 'lora_B2' in name:
            lora_B2 += param.flatten().tolist()
    all_samples = lora_B0 + lora_B1 + lora_B2
    samples_counts = [len(lora_B0),len(lora_B1),len(lora_B2)]
    all_samples = np.array(all_samples)
    all_samples = all_samples.reshape(-1,1)
    return all_samples,samples_counts
    # return np.array(all_samples),samples_counts


# Function to load and extract model parameters
def load_and_extract_params(model_paths):
    all_samples = []
    samples_counts = []
    for path in model_paths.values():
        model_params = load_file(path, device="cpu")
        samples = [param.data.cpu().numpy().flatten() for param in model_params.values()]
        all_samples += samples
        samples_counts.append(len(samples))
    return np.array(all_samples), samples_counts

# Ensure all paths are set
# if None in model_paths.values():
#     raise ValueError("One or more model paths are not set. Please check your environment variables.")

# Load and extract parameters from models
# all_samples, samples_counts = load_and_extract_params(model_paths)

model_path = 'results/finetune/043-hyper_lora-joint_all/checkpoint-1820/finetune_weights.bin'
all_samples, samples_counts = load_lora_params(model_path)

# Apply t-SNE
tsne = TSNE(n_components=1, init='pca', random_state=42)
transformed_params = tsne.fit_transform(all_samples)

# colors =['#0094C6','#F9A825','#20B2AA','#FF4E50', '#2ECC71'] #  Expanded or modified to fit the number of categories
colors =['#0094C6','#F9A825','#20B2AA']

# markers = ['o', 's', '^', '*', 'D'] #  Expanded or modified to fit the number of categories
markers = ['o', 's', '^']

with plt.style.context(['science', 'scatter']):
    start_idx = 0
    for idx, (category, count) in enumerate(zip(['lora_B0','lora_B1','lora_B2'], samples_counts)):
        end_idx = start_idx + count
        x = transformed_params[start_idx:end_idx, 0]
        y = transformed_params[start_idx:end_idx, 1]
        plt.scatter(x,y,
                    label=category, 
                    edgecolors='gray', 
                    facecolors=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)], 
                    linewidths=0.3, 
                    s=20)
        
        for i, (x_coord, y_coord) in enumerate(zip(x, y)):
            if i < 64: 
                plt.text(x_coord, y_coord, str((start_idx + i)%64), fontsize=9) 

        start_idx = end_idx

    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("LoRA_weight_layer.pdf")
    plt.show()

