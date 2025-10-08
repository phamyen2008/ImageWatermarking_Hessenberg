#
# -*- coding: utf-8 -*-att_Hess_new, final
import cv2
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
import os
import hashlib
import math
import time
import traceback

BLOCKSIZE = 4
WM_SIZE = 32
T = 0.04
# beta= 0.4
q = 20
ARNOLD_ITER = 10
# beta = 0.8
REDUNDANCY = 5
PRIVATE_KEY = "KB123"

def arnold_transform(image, iterations):
    n = image.shape[0]
    result = image.copy()
    for _ in range(iterations):
        new = np.zeros_like(result)
        for x in range(n):
            for y in range(n):
                new[(x + y) % n, (x + 2 * y) % n] = result[x, y]
        result = new
    return result

def invert_arnold_transform(image, iterations):
    n = image.shape[0]
    result = image.copy()
    for _ in range(iterations):
        new = np.zeros_like(result)
        for x in range(n):
            for y in range(n):
                new[(2 * x - y) % n, (-x + y) % n] = result[x, y]
        result = new
    return result

def generate_block_indices(img_shape, block_size, total_bits_needed, key):
    h, w = img_shape[:2]
    blocks_per_row = w // block_size
    blocks_per_col = h // block_size
    total_blocks = blocks_per_row * blocks_per_col
    indices = []
    used = set()
    i = 0
    md5_key = key.encode('utf-8')
    max_attempts = total_blocks * 10
    while len(indices) < total_bits_needed and i < max_attempts:
        data_to_hash = md5_key + str(i).encode('utf-8')
        md5_val = hashlib.md5(data_to_hash).hexdigest()
        idx = int(md5_val, 16) % total_blocks
        if idx not in used:
            used.add(idx)
            row = (idx // blocks_per_row) * block_size
            col = (idx % blocks_per_row) * block_size
            if row + block_size <= h and col + block_size <= w:
                indices.append((row, col))
        i += 1
    return indices

def embed(host_path, watermark_path, output_path, flag_path, arnold_iter=10):
    host_img_orig = cv2.imread(host_path)
    host_img = host_img_orig[:, :, 0].astype(np.float64)
    wm_gray = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    wm_gray = cv2.resize(wm_gray, (WM_SIZE, WM_SIZE), interpolation=cv2.INTER_NEAREST)
    wm_scrambled = arnold_transform(wm_gray, arnold_iter)
    wm_bits_orig = (wm_scrambled.flatten() > 127).astype(np.uint8)
    bits_to_embed = np.repeat(wm_bits_orig, REDUNDANCY)
    blocks = generate_block_indices(host_img.shape, BLOCKSIZE, len(bits_to_embed), PRIVATE_KEY)

    flag_list = []
    watermarked_channel = host_img.copy()

    for idx, (i, j) in enumerate(blocks):
        bit = bits_to_embed[idx]
        block = watermarked_channel[i:i+BLOCKSIZE, j:j+BLOCKSIZE].copy()
        try:
            H, Q = linalg.hessenberg(block, calc_q=True)

            # Phuong phap 1 - tren Q
            Q1 = Q.copy()
            q22, q32 = Q1[1, 1], Q1[2, 1]
            avgQ = (abs(q22) + abs(q32)) / 2
            # avgQ = (q22 + q32) / 2
            sq22, sq32 = np.sign(q22), np.sign(q32)

            if bit == 1:
                Q1[1, 1] = sq22 * (avgQ + T / 2)
                Q1[2, 1] = sq32 * (avgQ - T / 2)
            else:
                Q1[1, 1] = sq22 * (avgQ - T / 2)
                Q1[2, 1] = sq32 * (avgQ + T / 2)



            block1 = Q1 @ H @ Q1.T

            # Phuong phap 2 - tren H
            H2 = H.copy()
            h22 = H2[1, 1]
            z = h22 % q
            if bit == 0:
                if z < 3 * q / 4:
                    H2[1, 1] = h22 + (q / 4) - z
                else:
                    H2[1, 1] = h22 + (5 * q / 4) - z
            else:
                if z < q / 4:
                    H2[1, 1] = h22 - (q / 4) - z
                else:
                    H2[1, 1] = h22 + (3 * q / 4) - z

            block2 = Q @ H2 @ Q.T
            err1 = np.mean((block - block1)**2)
            err2 = np.mean((block - block2)**2)

            if err1 <= err2:
                watermarked_channel[i:i+BLOCKSIZE, j:j+BLOCKSIZE] = np.clip(block1, 0, 255)
                flag_list.append(0)
            else:
                watermarked_channel[i:i+BLOCKSIZE, j:j+BLOCKSIZE] = np.clip(block2, 0, 255)
                flag_list.append(1)

        except:
            continue

    watermarked_img_final = host_img_orig.copy()
    watermarked_img_final[:, :, 0] = np.round(watermarked_channel).astype(np.uint8)
    cv2.imwrite(output_path, watermarked_img_final)
    np.savetxt(flag_path, np.array(flag_list, dtype=np.uint8), fmt='%d')
    return watermarked_img_final

def extract(watermarked_path, flag_path, output_path, original_wm_shape=(WM_SIZE, WM_SIZE), arnold_iter=10):
    wm_img = cv2.imread(watermarked_path)
    wm_img_float = wm_img[:, :, 0].astype(np.float64)
    flags = np.loadtxt(flag_path, dtype=np.uint8).tolist()
    blocks = generate_block_indices(wm_img_float.shape, BLOCKSIZE, len(flags), PRIVATE_KEY)
    bit_votes = []
    extracted_bits = []

    for idx, (i, j) in enumerate(blocks):
        flag = flags[idx]
        block = wm_img_float[i:i+BLOCKSIZE, j:j+BLOCKSIZE]
        try:
            H, Q = linalg.hessenberg(block, calc_q=True)
            extracted_bit = 0
            if flag == 0:
                # Công thức trích xuất dựa trên ma trận Q
                q22a = abs(Q[1, 1])
                q32a = abs(Q[2, 1])
                extracted_bit = 1 if q22a > q32a else 0
            else:
                z = H[1, 1] % q
                extracted_bit = 1 if z >= q / 2 else 0
        except:
            extracted_bit = 0
        bit_votes.append(extracted_bit)
        if len(bit_votes) == REDUNDANCY:
            extracted_bits.append(1 if sum(bit_votes) > REDUNDANCY / 2 else 0)
            bit_votes = []

    final_bits = np.array(extracted_bits[:WM_SIZE*WM_SIZE]) * 255
    wm_extracted = final_bits.reshape(original_wm_shape).astype(np.uint8)
    descrambled = invert_arnold_transform(wm_extracted, arnold_iter)
    cv2.imwrite(output_path, descrambled)
    return descrambled

def calculate_metrics(original, watermarked, extracted_watermark, original_watermark_binary):
    psnr_value, ssim_value, nc_value = float('nan'), float('nan'), float('nan')

    # PSNR and SSIM (between original host and watermarked host)
    try:
        if original is None or watermarked is None:
            # print("Metrics Warning: Original or Watermarked image is None.")
            pass
        elif original.shape != watermarked.shape:
            # print(f"Metrics Warning: Shape mismatch between original ({original.shape}) and watermarked ({watermarked.shape}).")
            pass
        else:
            # --- PSNR ---
            # Chỉ tính trên kênh Blue (kênh 0) vì chỉ kênh đó bị thay đổi
            ori_f = original[:,:,0].astype(np.float64)
            wm_f = watermarked[:,:,0].astype(np.float64)
            mse = np.mean((ori_f - wm_f) ** 2)
            if mse == 0:
                psnr_value = float('inf') # Hoàn hảo, không có thay đổi
            else:
                max_pixel = 255.0
                psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))

            # --- SSIM ---
            # Tính SSIM trên ảnh grayscale hoặc từng kênh rồi lấy trung bình
            # Ở đây, tính trên kênh Blue (kênh 0)
            ori_u8_ch = original[:,:,0].astype(np.uint8)
            wm_u8_ch = watermarked[:,:,0].astype(np.uint8)
            # Đảm bảo win_size hợp lệ (lẻ và nhỏ hơn kích thước ảnh)
            min_dim = min(ori_u8_ch.shape)
            win_size = min(7, min_dim) # Giới hạn win_size tối đa là 7 hoặc kích thước nhỏ nhất
            win_size -= (1 - win_size % 2) # Đảm bảo là số lẻ

            if win_size >= 3: # win_size phải >= 3
                 # Sử dụng channel_axis=None vì đang làm việc với ảnh 2D (1 kênh)
                 ssim_value = ssim(ori_u8_ch, wm_u8_ch, data_range=255,
                                  win_size=win_size, gaussian_weights=True)
            else:
                 # print("Metrics Warning: Cannot calculate SSIM, window size too small.")
                 pass

    except Exception as e:
        print(f"Metrics Error (PSNR/SSIM): {e}")
        traceback.print_exc() # In chi tiết lỗi

    # NC (between original watermark and extracted watermark)
    try:
        if original_watermark_binary is None or extracted_watermark is None:
            # print("Metrics Warning: Original binary WM or Extracted WM is None.")
            pass
        elif original_watermark_binary.shape != extracted_watermark.shape:
             if extracted_watermark.size > 0:
                 extracted_watermark = cv2.resize(extracted_watermark,
                                                 (original_watermark_binary.shape[1], original_watermark_binary.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
                 print(f"Metrics Info: Resized extracted WM to {extracted_watermark.shape} for NC calculation.")
             else: pass # Không thể resize nếu rỗng

        # Proceed only if shapes match after potential resize
        if original_watermark_binary is not None and \
           extracted_watermark is not None and \
           original_watermark_binary.shape == extracted_watermark.shape:

             # Chuyển cả hai về dạng binary 0/1 hoặc float 0.0/1.0 để tính toán NC
             ori_wm_flat = (original_watermark_binary.flatten() > 127).astype(np.float64)
             ext_wm_flat = (extracted_watermark.flatten() > 127).astype(np.float64)

             # Kiểm tra xem vector có toàn 0 không
             sum_ori_sq = np.sum(ori_wm_flat**2)
             sum_ext_sq = np.sum(ext_wm_flat**2)

             if sum_ori_sq == 0 or sum_ext_sq == 0:
                 nc_value = 1.0 if sum_ori_sq == sum_ext_sq else 0.0
             else:
                 # Tính toán NC
                 numerator = np.sum(ori_wm_flat * ext_wm_flat)
                 denominator = np.sqrt(sum_ori_sq * sum_ext_sq)
                 nc_value = numerator / denominator if denominator != 0 else 0.0
        else:
            # print("Metrics Info: NC not calculated due to prior warnings or shape mismatch.")
            pass

    except Exception as e:
        print(f"Metrics Error (NC): {e}")
        traceback.print_exc() # In chi tiết lỗi

    return psnr_value, ssim_value, nc_value

def process_images(arnold_iter=10):
    input_folder = "Image512"
    watermark_path = os.path.join(input_folder, "w_binary.png")

    # Đọc và chuẩn bị watermark gốc (đảm bảo đúng kích thước và nhị phân)
    original_watermark_raw = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if original_watermark_raw is None:
        print(f"CRITICAL Error: Cannot read original watermark file: {watermark_path}. Exiting.")
        return
    if original_watermark_raw.shape != (WM_SIZE, WM_SIZE):
        original_watermark_resized = cv2.resize(original_watermark_raw, (WM_SIZE, WM_SIZE), interpolation=cv2.INTER_NEAREST)
    else:
        original_watermark_resized = original_watermark_raw
    # Đảm bảo watermark gốc là ảnh nhị phân (0 và 255) để tính NC chính xác
    _, original_watermark_binary = cv2.threshold(original_watermark_resized, 127, 255, cv2.THRESH_BINARY)


    # Cập nhật tên phiên bản để phản ánh việc loại bỏ ECC
    version_name = "hess_new" # Ví dụ: v5, no_ecc
    output_folder_img = f"output_{version_name}_img"
    output_folder_wm = f"output_{version_name}_wm"
    output_folder_flag = f"output_{version_name}_flag"
    os.makedirs(output_folder_img, exist_ok=True)
    os.makedirs(output_folder_wm, exist_ok=True)
    os.makedirs(output_folder_flag, exist_ok=True)


    results = {}
    valid_extensions = ('.bmp') # Mở rộng định dạng ảnh
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions) and not f.startswith('w_')]


    for filename in image_files:
        image_path = os.path.join(input_folder, filename)
        original_img = cv2.imread(image_path) # Đọc ảnh gốc để tính metric
        if original_img is None:
            print(f"Skip {filename}: read error.")
            continue

        # Chỉ lấy kênh Blue của ảnh gốc để tính PSNR/SSIM nếu muốn so sánh đúng kênh đã sửa
        # original_img_blue_channel = original_img[:,:,0]

        print(f"\n--- Processing: {filename} ---")
        base_filename = os.path.splitext(filename)[0]
        watermarked_path = os.path.join(output_folder_img, f"watermarked_{base_filename}.png") # Luôn lưu png để tránh mất mát
        flag_path = os.path.join(output_folder_flag, f"flag_{base_filename}.txt")
        extracted_wm_path = os.path.join(output_folder_wm, f"extracted_{base_filename}.png")

        watermarked_img, extracted_watermark = None, None
        embed_time, extract_time = 0, 0

        # Embedding
        print(f"Embedding...")
        start_time = time.time()
        try:
            # Hàm embed giờ trả về ảnh 3 kênh đã watermarked
            watermarked_img = embed(image_path, watermark_path, watermarked_path, flag_path, arnold_iter)
        except Exception as e:
            print(f"Embedding failed for {filename}: {e}")
            traceback.print_exc() # In chi tiết lỗi
            # Nếu nhúng lỗi, không cần cố gắng trích xuất
            watermarked_img = None # Đảm bảo watermarked_img là None nếu lỗi
        embed_time = time.time() - start_time

      
        start_time = time.time()
        if watermarked_img is not None and os.path.exists(flag_path):
             print(f"Extracting...")
             try:
                 extracted_watermark = extract(watermarked_path, flag_path, extracted_wm_path, (WM_SIZE, WM_SIZE), arnold_iter)
             except Exception as e:
                 print(f"Extraction failed for {filename}: {e}")
                 traceback.print_exc() # In chi tiết lỗi
                 extracted_watermark = None # Đặt là None nếu trích xuất lỗi
        elif watermarked_img is None:
             print(f"Skipping extraction: Embedding failed for {filename}.")
        else: # watermarked_img is not None nhưng flag file không tồn tại
             print(f"Skipping extraction: Flag file not found at {flag_path} (Embedding might have failed partially or file was deleted).")
        extract_time = time.time() - start_time # Tính thời gian ngay cả khi bỏ qua hoặc lỗi


        # Calculate Metrics
        print("Calculating metrics...")

        psnr_val, ssim_val, nc_val = calculate_metrics(original_img, watermarked_img, extracted_watermark, original_watermark_binary)

        results[filename] = {'PSNR': psnr_val, 'SSIM': ssim_val, 'NC': nc_val, 'EmbedTime': embed_time, 'ExtractTime': extract_time}
        print(f"Metrics for {filename}: PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, NC: {nc_val:.4f}, Embed: {embed_time:.4f}s, Extract: {extract_time:.4f}s")






if __name__ == "__main__":

    # try: import bchlib
    # except ImportError: print("Error: 'bchlib' required. Install via 'pip install bchlib'"); exit()

    process_images() # Gọi hàm xử lý chính
    print("\nAll processing finished.")