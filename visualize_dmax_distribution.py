import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import linalg

BLOCKSIZE = 4

def get_d_elements_distribution(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return []

    d_elements_indices = []

    for y in range(0, img.shape[0], BLOCKSIZE):
        for x in range(0, img.shape[1], BLOCKSIZE):
            block = img[y:y+BLOCKSIZE, x:x+BLOCKSIZE]
            if block.shape[0] == BLOCKSIZE and block.shape[1] == BLOCKSIZE:
                try:
                    D, _ = linalg.schur(block.astype(np.float64))
                    max_index = np.unravel_index(np.argmax(D, axis=None), D.shape)
                    d_elements_indices.append(max_index)
                except linalg.LinAlgError:
                    continue
    
    return d_elements_indices

def visualize_d_elements_distribution(d_elements_indices, total_blocks, image_name):
    unique_indices, counts = np.unique(d_elements_indices, return_counts=True, axis=0)
    percentages = 100 * counts / total_blocks
    
    plt.figure(figsize=(10, 8))
    wedges, _ = plt.pie(counts, startangle=140, colors=plt.cm.Paired.colors[:len(unique_indices)])
    plt.legend(wedges, [f"Index {idx}: {count} blocks ({percent:.1f}%)" for idx, count, percent in zip(unique_indices, counts, percentages)], title="Indices", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title(f"Distribution of Indices of Maximum D Elements\nImage: {image_name}, Total Blocks: {total_blocks}")
    plt.show()

def process_image(image_path):
    d_elements_indices = get_d_elements_distribution(image_path)
    total_blocks = calculate_total_blocks((256, 256), (4, 4))  
    image_name = os.path.basename(image_path) 
    if d_elements_indices:
        visualize_d_elements_distribution(d_elements_indices, total_blocks, image_name)
    else:
        print("No valid data processed.")

def calculate_total_blocks(image_size, block_size):
    blocks_per_row = image_size[0] // block_size[0]
    blocks_per_column = image_size[1] // block_size[1]
    total_blocks = blocks_per_row * blocks_per_column
    return total_blocks

image_path = "image512/tiffany.bmp"
process_image(image_path)