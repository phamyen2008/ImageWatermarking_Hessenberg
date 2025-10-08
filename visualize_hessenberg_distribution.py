import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import hessenberg

BLOCKSIZE = 4


def get_hessenberg_elements_distribution(img):
    h_elements_indices = []
    for y in range(0, img.shape[0], BLOCKSIZE):
        for x in range(0, img.shape[1], BLOCKSIZE):
            block = img[y:y + BLOCKSIZE, x:x + BLOCKSIZE]
            if block.shape[0] == BLOCKSIZE and block.shape[1] == BLOCKSIZE:
                try:
                    H, _ = hessenberg(block.astype(np.float64), calc_q=True)
                    max_index = np.unravel_index(np.argmax(H, axis=None), H.shape)
                    h_elements_indices.append(max_index)
                except np.linalg.LinAlgError:
                    continue
    return h_elements_indices


def analyze_hessenberg_max_values(img, t_value=65):
    h_max_values = []
    h_max_modulo_values = []

    for y in range(0, img.shape[0], BLOCKSIZE):
        for x in range(0, img.shape[1], BLOCKSIZE):
            block = img[y:y + BLOCKSIZE, x:x + BLOCKSIZE]
            if block.shape[0] == BLOCKSIZE and block.shape[1] == BLOCKSIZE:
                try:
                    H, _ = hessenberg(block.astype(np.float64), calc_q=True)
                    max_value = np.max(H)
                    h_max_values.append(max_value)
                    h_max_modulo_values.append(max_value % t_value)
                except np.linalg.LinAlgError:
                    continue
    return h_max_values, h_max_modulo_values


def calculate_total_blocks(image_size, block_size):
    blocks_per_row = image_size[0] // block_size[0]
    blocks_per_column = image_size[1] // block_size[1]
    total_blocks = blocks_per_row * blocks_per_column
    return total_blocks


def visualize_all_images(folder_path):
    bmp_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.bmp')]
    if not bmp_files:
        print("Không có ảnh .bmp nào trong thư mục!")
        return

    # Tính số hàng cần thiết (3 ảnh một hàng)
    n_images = len(bmp_files)
    n_rows = (n_images + 2) // 3  # Làm tròn lên

    # Tạo figure với n_rows hàng, 3 cột
    fig, axs = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)

    for idx, fname in enumerate(bmp_files):
        row = idx // 3
        col = idx % 3

        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue

        # Lấy dữ liệu phân tích
        h_elements_indices = get_hessenberg_elements_distribution(img)
        h_max_values, h_max_modulo_values = analyze_hessenberg_max_values(img)

        if h_elements_indices:
            # Vẽ biểu đồ phân bố chỉ số max H
            unique_indices, counts = np.unique(h_elements_indices, return_counts=True, axis=0)
            total_blocks = calculate_total_blocks(img.shape, (BLOCKSIZE, BLOCKSIZE))
            percentages = 100 * counts / total_blocks

            wedges, _ = axs[row, col].pie(counts, startangle=140, colors=plt.cm.Paired.colors[:len(unique_indices)])
            axs[row, col].set_title(f"{fname}\nDistribution of Max H Indices\nTotal: {total_blocks} blocks")

            # Thêm legend nếu có ít hơn 5 loại
            if len(unique_indices) <= 5:
                legend_labels = [f"Index {idx}: {count} ({percent:.1f}%)"
                                 for idx, count, percent in zip(unique_indices, counts, percentages)]
                axs[row, col].legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))
        else:
            axs[row, col].text(0.5, 0.5, f"No data for {fname}", ha='center', va='center',
                               transform=axs[row, col].transAxes)
            axs[row, col].set_title(fname)

    # Ẩn các subplot không sử dụng
    for idx in range(n_images, n_rows * 3):
        row = idx // 3
        col = idx % 3
        axs[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder_path = "Image512"
    visualize_all_images(folder_path)