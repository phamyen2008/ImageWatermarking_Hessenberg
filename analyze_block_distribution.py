import numpy as np
import matplotlib.pyplot as plt
import os
import json

folder_path = ''  # Folder containing flag files
flag_files = [f for f in os.listdir(folder_path) if f.endswith(('.json', '.txt'))]
num_files = len(flag_files)
cols = 4
rows = (num_files + cols - 1) // cols

plt.figure(figsize=(cols*5, rows*5))
for idx, flag_file in enumerate(flag_files):
    flag_path = os.path.join(folder_path, flag_file)
    
    # Đọc dữ liệu từ file
    if flag_file.endswith('.json'):
        with open(flag_path, 'r') as f:
            flags = np.array(json.load(f), dtype=np.uint8)
    elif flag_file.endswith('.txt'):
        with open(flag_path, 'r') as f:
            flags = np.array([int(line.strip()) for line in f if line.strip()], dtype=np.uint8)
    
    method_u_count = np.sum(flags == 0)
    method_d_count = np.sum(flags == 1)
    total_blocks = len(flags)
    method_u_percentage = (method_u_count / total_blocks) * 100
    method_d_percentage = (method_d_count / total_blocks) * 100
    labels = ['Method Q', 'Method H']
    percentages = [method_u_percentage, method_d_percentage]
    ax = plt.subplot(rows, cols, idx+1)
    ax.pie(percentages, labels=labels, autopct='%1.1f%%')
    ax.set_title(flag_file)
plt.tight_layout()
plt.show()