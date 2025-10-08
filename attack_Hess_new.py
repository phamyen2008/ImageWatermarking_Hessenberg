# Hess_new/Schur_new
import cv2
import numpy as np
import os
import time
from PIL import Image
import io

from scipy.ndimage import median_filter

from Schur_new import embed, extract, calculate_metrics  # Import from su_new
from scipy.fftpack import dct, idct
from wand.image import Image

def clear_screen():
    os.system('cls')

def scale_attack(image, scale_factor):
    h, w = image.shape[:2]
    scaled = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
    return cv2.resize(scaled, (w, h))

def salt_pepper_attack(image, noise_density):
    noisy_img = image.copy()
    salt_pepper_noise = np.zeros_like(image)
    salt_pepper_pixels = np.random.rand(image.shape[0], image.shape[1])
    
    salt_pepper_noise[salt_pepper_pixels < noise_density/2] = 0
    salt_pepper_noise[salt_pepper_pixels > 1 - noise_density/2] = 255
    
    noisy_img = cv2.add(image, salt_pepper_noise)
    return noisy_img

def rotation_attack(image, angle):
    center = tuple(np.array(image.shape[0:2])/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape[0:2], flags=cv2.INTER_LINEAR)
    rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated_image = cv2.warpAffine(rotated_image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
    return rotated_image


def gaussian_noise_attack(image, mean, std):
    # Chuẩn hóa về [0,1]
    img_float = image.astype(np.float32) / 255.0
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_img = np.clip(img_float + noise, 0.0, 1.0)
    return (noisy_img * 255).astype(np.uint8)

# def gaussian_noise_attack(image, mean, variance):
#     """
#     Thêm nhiễu Gaussian vào ảnh.
#
#     Parameters:
#     - image: Ảnh đầu vào (numpy array).
#     - variance: Phương sai của nhiễu (giá trị càng lớn thì nhiễu càng nhiều).
#
#     Returns:
#     - Ảnh đã bị nhiễu Gaussian.
#     """
#     # Tính độ lệch chuẩn từ phương sai
#     sigma = variance ** 0.5
#
#     # Tạo Gaussian noise với trung bình 0 và độ lệch chuẩn sigma
#     gauss = np.random.normal(0, sigma, image.shape).astype(np.float32)
#
#     # Chuyển ảnh sang kiểu float32 để cộng noise
#     noisy_image = image.astype(np.float32) / 255.0
#
#     # Thêm nhiễu vào ảnh
#     noisy_image += gauss
#
#     # Giới hạn giá trị trong khoảng [0,1]
#     noisy_image = np.clip(noisy_image, 0, 1)
#
#     # Chuyển về định dạng uint8 (0-255)
#     noisy_image = (noisy_image * 255).astype(np.uint8)
#
#     return noisy_image




def blur_attack(image, const):
    return cv2.GaussianBlur(image, (0, 0), const)

def compress_jpeg_attack(image, quality_factor):
    temp_input = 'temp_input.bmp'
    temp_output = 'temp_output.jpg'
    cv2.imwrite(temp_input, image)

    with Image(filename=temp_input) as img:
        img.compression_quality = quality_factor
        img.save(filename=temp_output)

    attacked_img = cv2.imread(temp_output)
    os.remove(temp_input)
    os.remove(temp_output)

    return attacked_img


    # --- Decode về ảnh ---
    flags = cv2.IMREAD_COLOR if keep_color else cv2.IMREAD_GRAYSCALE
    attacked = cv2.imdecode(enc, flags)
    if attacked is None:
        # Thử phương án phụ khi imdecode lỗi hiếm gặp
        data = np.frombuffer(enc.tobytes() if isinstance(enc, np.ndarray) else enc, dtype=np.uint8)
        attacked = cv2.imdecode(data, flags)

    if attacked is None:
        raise RuntimeError("cv2.imdecode() failed to decode JPEG buffer.")

    return attacked


# def median_filter_attack(image, kernel_size):
#     if kernel_size % 2 == 0:
#         kernel_size += 1
#     image = np.clip(image, 0, 255).astype(np.uint8)
#     filtered = cv2.medianBlur(image, kernel_size)
#     return filtered

def median_filter_attack(image, kernel_size):
    if image.ndim == 2:
        # Grayscale
        return median_filter(image, size=(kernel_size, kernel_size), mode='reflect')
    elif image.ndim == 3:
        # RGB
        return np.stack([
            median_filter(image[:, :, c], size=(kernel_size, kernel_size), mode='reflect')
            for c in range(image.shape[2])
        ], axis=2)
    else:
        raise ValueError("Unsupported image dimensions")


def histogram_attack(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    else:
        return cv2.equalizeHist(image)

def lowpass_filter_attack(image, kernel_size_x, kernel_size_y):
    # Cho phép kernel hình chữ nhật (ví dụ (100,1) hoặc (100,3))
    if kernel_size_x % 2 == 0:
        kernel_size_x += 1
    if kernel_size_y % 2 == 0:
        kernel_size_y += 1
    return cv2.GaussianBlur(image, (kernel_size_x, kernel_size_y), 0)

# def jp2_attack(image, compression_ratio):
#     quality_layer = int(max(1, 100 - compression_ratio))
#     params = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), quality_layer]
#     result, encimg = cv2.imencode('.jp2', image, params)
#     attacked_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
#     return attacked_img

# def jp2_attack(image, compression_ratio):
#     """Tấn công nén JPEG2000"""
#     compression_x1000 = int(compression_ratio * 1000)
#     params = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), compression_x1000]
#     result, encimg = cv2.imencode('.jp2', image, params)
#
#     if len(image.shape) == 3:
#         attacked_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
#     else:
#         attacked_img = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
#
#     return attacked_img

def jp2_attack(image, compression_ratio):
    """
    Tấn công JPEG2000 với compression_ratio ∈ [1, 100]
    - compression_ratio = 10 → compression_x1000 = 100
    - compression_ratio = 80 → compression_x1000 = 800
    """
    compression_ratio = max(1, min(compression_ratio, 100))  # Giới hạn [1, 100]
    compression_x1000 = int(compression_ratio * 10)  # chuyển về [10, 1000]

    params = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), compression_x1000]
    result, encimg = cv2.imencode('.jp2', image, params)

    if not result:
        raise ValueError("Không thể mã hóa ảnh JPEG2000")

    if len(image.shape) == 3:
        attacked_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    else:
        attacked_img = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)

    return attacked_img


def sharpen_attack(image, const):
    alpha = 1.5
    beta = -0.5
    gamma = 0
    blurred_image = cv2.GaussianBlur(image, (0, 0), const)
    sharpened_image = cv2.addWeighted(image, alpha, blurred_image, beta, gamma)
    return sharpened_image

def crop_attack(image, crop_percent, direction=None, corner=None):
    height, width = image.shape[:2]
    img_cropped = image.copy()
    
    if crop_percent < 50:
        half_height = height // 2
        half_width = width // 2
        
        if corner:
            crop_solid = np.zeros((half_height, half_width, 3), dtype=np.uint8)  
            if corner == 'top-left':
                img_cropped[:half_height, :half_width] = crop_solid
            elif corner == 'top-right':
                img_cropped[:half_height, half_width:] = crop_solid
            elif corner == 'bottom-left':
                img_cropped[half_height:, :half_width] = crop_solid
            elif corner == 'bottom-right':
                img_cropped[half_height:, half_width:] = crop_solid
        else:
            if direction in ['left', 'right']:
                crop_width = int(width * crop_percent / 100)
                crop_solid = np.zeros((height, crop_width, 3), dtype=np.uint8)
                if direction == 'left':
                    img_cropped[:, :crop_width] = crop_solid
                else:
                    img_cropped[:, -crop_width:] = crop_solid
            else:
                crop_height = int(height * crop_percent / 100)
                crop_solid = np.zeros((crop_height, width, 3), dtype=np.uint8)
                if direction == 'top':
                    img_cropped[:crop_height, :] = crop_solid
                else:
                    img_cropped[-crop_height:, :] = crop_solid
    else:
        crop_height = int(height * crop_percent / 100)
        crop_width = int(width * crop_percent / 100)
        
        if direction in ['left', 'right']:
            crop_solid = np.zeros((height, crop_width, 3), dtype=np.uint8)
            if direction == 'left':
                img_cropped[:, :crop_width] = crop_solid
            else:
                img_cropped[:, -crop_width:] = crop_solid
        else:
            crop_solid = np.zeros((crop_height, width, 3), dtype=np.uint8)
            if direction == 'top':
                img_cropped[:crop_height, :] = crop_solid
            else:
                img_cropped[-crop_height:, :] = crop_solid

    return img_cropped 
