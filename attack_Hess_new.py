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

def average_filter_attack(image, param):
    # param là kích thước kernel, ví dụ 7
    return cv2.blur(image, (param, param))

def translation_attack(image, shift_x, shift_y):
    height, width = image.shape[:2]
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    return translated_image

def process_attack(input_folder, watermark_path, image_name, attack_type, attack_params=None):
    base_dir = "Schur_new_attacks"
    attack_folder = os.path.join(base_dir, attack_type.lower().replace(' ', '_'))
    os.makedirs(attack_folder, exist_ok=True)

    img_path = os.path.join(input_folder, image_name)
    host_image = cv2.imread(img_path)
    if host_image is None:
        print(f"Error: Could not read image {image_name}")
        return

    print(f"\nProcessing {image_name} with {attack_type}...")
    start_time = time.time()
    
    watermarked_path = os.path.join(attack_folder, f"watermarked_{image_name}")
    flag_path = os.path.join(attack_folder, f"flag_{image_name}.txt")
    watermarked_img = embed(img_path, watermark_path, watermarked_path, flag_path)
    if watermarked_img is None:
        return

    try:
        if attack_type == "Scale":
            if attack_params is None:
                scale_input = input("Enter scale factor (e.g., 0.5, 2.0): ")
                scale = float(scale_input)
            else:
                scale = attack_params
            attacked_img = scale_attack(watermarked_img, scale)
            param_str = f"_scale{scale}"
        
        elif attack_type == "Salt & Pepper":
            if attack_params is None:
                density_input = input("Enter noise density (0-1): ")
                density = float(density_input)
            else:
                density = attack_params
            attacked_img = salt_pepper_attack(watermarked_img, density)
            param_str = f"_density{density}"
        
        elif attack_type == "Gaussian Noise":
            if attack_params is None:
                mean_input = input("Enter mean (0-255): ")
                mean = float(mean_input)
                std_input = input("Enter standard deviation: ")
                std = float(std_input)
            else:
                mean, std = attack_params
            attacked_img = gaussian_noise_attack(watermarked_img, mean, std)
            param_str = f"_mean{mean}_std{std}"
        
        elif attack_type == "Blur":
            if attack_params is None:
                const_input = input("Enter blur constant (e.g., 0.4): ")
                const = float(const_input)
            else:
                const = attack_params
            attacked_img = blur_attack(watermarked_img, const)
            param_str = f"_const{const}"
        
        elif attack_type == "JPEG":
            if attack_params is None:
                print("\nSelect quality factor:")
                print("1. QF = 50 ")
                print("2. QF = 70 ")
                print("3. QF = 90 ")
                qf_choice_input = input("Enter choice (1-3): ")
                qf_choice = int(qf_choice_input)
                quality_factors = [50, 70, 90]
                quality_factor = quality_factors[qf_choice - 1]
            else:
                quality_factor = attack_params
            attacked_img = compress_jpeg_attack(watermarked_img, quality_factor)
            param_str = f"_qf{quality_factor}"
        
        elif attack_type == "Rotation":
            if attack_params is None:
                angle_input = input("Enter rotation angle (degrees): ")
                angle = float(angle_input)
            else:
                angle = attack_params
            attacked_img = rotation_attack(watermarked_img, angle)
            param_str = f"_angle{angle}"
        
        elif attack_type == "Median":
            if attack_params is None:
                kernel_input = input("Enter kernel size: ")
                kernel = int(kernel_input)
            else:
                kernel = attack_params
            attacked_img = median_filter_attack(watermarked_img, kernel)
            param_str = f"_kernel{kernel}"

        elif attack_type == "Translation":
            if attack_params is None:
                shift_x_input = input("Enter horizontal shift (pixels): ")
                shift_x = int(shift_x_input)
                shift_y_input = input("Enter vertical shift (pixels): ")
                shift_y = int(shift_y_input)
            else:
                shift_x, shift_y = attack_params
            attacked_img = translation_attack(watermarked_img, shift_x, shift_y)
            param_str = f"_x{shift_x}y{shift_y}"
        
        elif attack_type == "Histogram":
            attacked_img = histogram_attack(watermarked_img)
            param_str = "_hist"
        
        elif attack_type == "Lowpass Filter":
            if attack_params is None:
                print("Enter kernel size (e.g., 100 1 or 100 3):")
                kernel_x_input = input("Kernel size X: ")
                kernel_y_input = input("Kernel size Y: ")
                kernel_x = int(kernel_x_input)
                kernel_y = int(kernel_y_input)
            else:
                kernel_x, kernel_y = attack_params
            attacked_img = lowpass_filter_attack(watermarked_img, kernel_x, kernel_y)
            param_str = f"_kernel{kernel_x}x{kernel_y}"
        
        elif attack_type == "JPEG2000":
            if attack_params is None:
                compression_ratio_input = input("Enter compression ratio (1-100): ")
                compression_ratio = float(compression_ratio_input)
            else:
                compression_ratio = attack_params
            attacked_img = jp2_attack(watermarked_img, compression_ratio)
            param_str = f"_cr{compression_ratio}"
        
        elif attack_type == "Sharpen":
            if attack_params is None:
                const_input = input("Enter sharpening constant (recommended > 1): ")
                const = float(const_input)
            else:
                const = attack_params
            attacked_img = sharpen_attack(watermarked_img, const)
            param_str = f"_const{const}"
        
        elif attack_type == "Crop":
            if attack_params is None:
                crop_percent_input = input("Enter crop percentage (0-100): ")
                crop_percent = float(crop_percent_input)
                if crop_percent < 50:
                    print("\nSelect cropping type:")
                    print("1. Corner cropping")
                    print("2. Direction cropping")
                    crop_type_input = input("Enter choice (1-2): ")
                    crop_type = int(crop_type_input)
                    
                    if crop_type == 1:
                        print("\nSelect corner to crop from:")
                        print("1. Top-Left")
                        print("2. Top-Right")
                        print("3. Bottom-Left")
                        print("4. Bottom-Right")
                        corner_choice_input = input("Enter choice (1-4): ")
                        corner_choice = int(corner_choice_input)
                        corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
                        corner = corners[corner_choice - 1]
                        attacked_img = crop_attack(watermarked_img, crop_percent, corner=corner)
                        param_str = f"_crop{crop_percent}_{corner}"
                    else:
                        print("\nSelect cropping direction:")
                        print("1. Top")
                        print("2. Bottom")
                        print("3. Left")
                        print("4. Right")
                        direction_choice_input = input("Enter choice (1-4): ")
                        direction_choice = int(direction_choice_input)
                        directions = ['top', 'bottom', 'left', 'right']
                        direction = directions[direction_choice - 1]
                        attacked_img = crop_attack(watermarked_img, crop_percent, direction=direction)
                        param_str = f"_crop{crop_percent}_{direction}"
                else:
                    print("\nSelect cropping direction:")
                    print("1. Top")
                    print("2. Bottom")
                    print("3. Left")
                    print("4. Right")
                    direction_choice_input = input("Enter choice (1-4): ")
                    direction_choice = int(direction_choice_input)
                    directions = ['top', 'bottom', 'left', 'right']
                    direction = directions[direction_choice - 1]
                    attacked_img = crop_attack(watermarked_img, crop_percent, direction=direction)
                    param_str = f"_crop{crop_percent}_{direction}"
            else:
                crop_percent, direction = attack_params
                attacked_img = crop_attack(watermarked_img, crop_percent, direction=direction)
                param_str = f"_crop{crop_percent}_{direction}"
        
        elif attack_type == "Average Filter":
            if attack_params is None:
                param_input = input("Enter kernel size (e.g., 7): ")
                param = int(param_input)
            else:
                param = attack_params
            attacked_img = average_filter_attack(watermarked_img, param)
            param_str = f"_avg{param}"

        attack_name = attack_type.lower().replace(' ', '_').replace('&', 'and')
        attacked_folder = os.path.join(attack_folder, f"{attack_name}_images")
        extracted_folder = os.path.join(attack_folder, "extracted_watermarks")
        os.makedirs(attacked_folder, exist_ok=True)
        os.makedirs(extracted_folder, exist_ok=True)

        base_name = os.path.splitext(image_name)[0]
        attacked_path = os.path.join(attacked_folder, f"{attack_name}_{base_name}{param_str}.bmp")
        cv2.imwrite(attacked_path, attacked_img)

        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        original_extracted = extract(watermarked_path, flag_path,
                                os.path.join(extracted_folder, f"original_{image_name}"))
        extracted_watermark = extract(attacked_path, flag_path,
                                    os.path.join(extracted_folder, f"extracted_{base_name}{param_str}.png"))

        if original_extracted is not None and extracted_watermark is not None:
            psnr_orig, ssim_orig, nc_orig = calculate_metrics(
                host_image, watermarked_img, original_extracted, watermark
            )
            psnr_val, ssim_val, nc_val = calculate_metrics(
                host_image, attacked_img, extracted_watermark, watermark
            )

            print(f"\n{image_name} - Original Metrics:")
            print("---------------------------")
            print(f"PSNR: {psnr_orig:.2f}")
            print(f"SSIM: {ssim_orig:.4f}")
            print(f"NC: {nc_orig:.4f}")

            print(f"\n{image_name} - {attack_type} Results:")
            print("---------------------------")
            print(f"PSNR: {psnr_val:.2f}")
            print(f"SSIM: {ssim_val:.4f}")
            print(f"NC: {nc_val:.4f}")
            print(f"NC degradation: {nc_orig - nc_val:.4f}")

        total_time = time.time() - start_time
        print(f"\nTotal Processing Time for {image_name}: {total_time:.4f} seconds")
        
        return {
            'image_name': image_name,
            'psnr_orig': psnr_orig,
            'ssim_orig': ssim_orig,
            'nc_orig': nc_orig,
            'psnr_attacked': psnr_val,
            'ssim_attacked': ssim_val,
            'nc_attacked': nc_val,
            'nc_degradation': nc_orig - nc_val,
            'processing_time': total_time
        }

    except ValueError as e:
        print(f"\nError: Invalid input - {str(e)}")
        return None
    except Exception as e:
        print(f"\nError: {str(e)}")
        return None

def process_batch_attack(input_folder, watermark_path, images, attack_type, attack_params):
    """Xử lý tấn công hàng loạt với cùng một loại tấn công và tham số"""
    results = []
    total_start_time = time.time()
    
    print(f"\n=== Bắt đầu tấn công hàng loạt: {attack_type} ===")
    print(f"Số lượng ảnh: {len(images)}")
    print(f"Tham số tấn công: {attack_params}")
    
    for i, image_name in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Đang xử lý: {image_name}")
        result = process_attack(input_folder, watermark_path, image_name, attack_type, attack_params)
        if result:
            results.append(result)
    
    total_time = time.time() - total_start_time
    
    # In báo cáo tổng hợp
    print(f"\n=== BÁO CÁO TỔNG HỢP ===")
    print(f"Loại tấn công: {attack_type}")
    print(f"Tham số: {attack_params}")
    print(f"Tổng số ảnh xử lý: {len(results)}")
    print(f"Tổng thời gian: {total_time:.2f} giây")
    print(f"Thời gian trung bình mỗi ảnh: {total_time/len(results):.2f} giây")
    
    if results:
        avg_nc_degradation = sum(r['nc_degradation'] for r in results) / len(results)
        min_nc_degradation = min(r['nc_degradation'] for r in results)
        max_nc_degradation = max(r['nc_degradation'] for r in results)
        
        print(f"\nThống kê NC degradation:")
        print(f"Trung bình: {avg_nc_degradation:.4f}")
        print(f"Tối thiểu: {min_nc_degradation:.4f}")
        print(f"Tối đa: {max_nc_degradation:.4f}")
    
    # Giữ nguyên kết quả trên console
    print(f"\n" + "="*60)
    print("🎯 TẤN CÔNG HOÀN THÀNH!")
    print(" Kết quả đã được lưu trong thư mục su_new_attacks/")
    print(" Các file ảnh đã tấn công và watermark đã trích xuất đã được lưu")
    print("="*60)
    
    return results

def get_attack_parameters(attack_type):
    """Lấy tham số tấn công từ người dùng"""
    if attack_type == "Scale":
        scale_input = input("Nhập hệ số scale (ví dụ: 0.5, 2.0): ")
        return float(scale_input)
    
    elif attack_type == "Salt & Pepper":
        density_input = input("Nhập mật độ nhiễu (0-1): ")
        return float(density_input)
    
    elif attack_type == "Gaussian Noise":
        mean_input = input("Nhập mean (0-255): ")
        std_input = input("Nhập standard deviation: ")
        return (float(mean_input), float(std_input))
    
    elif attack_type == "Blur":
        const_input = input("Nhập hằng số blur (ví dụ: 0.4): ")
        return float(const_input)
    
    elif attack_type == "JPEG":
        print("\nChọn quality factor:")
        print("1. QF = 50")
        print("2. QF = 70")
        print("3. QF = 90")
        qf_choice = int(input("Nhập lựa chọn (1-3): "))
        quality_factors = [50, 70, 90]
        return quality_factors[qf_choice - 1]
    
    elif attack_type == "Rotation":
        angle_input = input("Nhập góc xoay (độ): ")
        return float(angle_input)
    
    elif attack_type == "Median":
        kernel_input = input("Nhập kích thước kernel: ")
        return int(kernel_input)
    
    elif attack_type == "Translation":
        shift_x_input = input("Nhập dịch chuyển ngang (pixels): ")
        shift_y_input = input("Nhập dịch chuyển dọc (pixels): ")
        return (int(shift_x_input), int(shift_y_input))
    
    elif attack_type == "Histogram":
        return None
    
    elif attack_type == "Lowpass Filter":
        print("Nhập kích thước kernel (ví dụ: 100 1 hoặc 100 3):")
        kernel_x = int(input("Kernel size X: "))
        kernel_y = int(input("Kernel size Y: "))
        return (kernel_x, kernel_y)
    
    elif attack_type == "JPEG2000":
        compression_ratio_input = input("Nhập tỷ lệ nén (1-100): ")
        return float(compression_ratio_input)
    
    elif attack_type == "Sharpen":
        const_input = input("Nhập hằng số sharpening (khuyến nghị > 1): ")
        return float(const_input)
    
    elif attack_type == "Crop":
        crop_percent_input = input("Nhập phần trăm crop (0-100): ")
        crop_percent = float(crop_percent_input)
        if crop_percent < 50:
            print("\nChọn loại crop:")
            print("1. Corner cropping")
            print("2. Direction cropping")
            crop_type = int(input("Nhập lựa chọn (1-2): "))
            
            if crop_type == 1:
                print("\nChọn góc để crop:")
                print("1. Top-Left")
                print("2. Top-Right")
                print("3. Bottom-Left")
                print("4. Bottom-Right")
                corner_choice = int(input("Nhập lựa chọn (1-4): "))
                corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
                corner = corners[corner_choice - 1]
                return (crop_percent, corner)
            else:
                print("\nChọn hướng crop:")
                print("1. Top")
                print("2. Bottom")
                print("3. Left")
                print("4. Right")
                direction_choice = int(input("Nhập lựa chọn (1-4): "))
                directions = ['top', 'bottom', 'left', 'right']
                direction = directions[direction_choice - 1]
                return (crop_percent, direction)
        else:
            print("\nChọn hướng crop:")
            print("1. Top")
            print("2. Bottom")
            print("3. Left")
            print("4. Right")
            direction_choice = int(input("Nhập lựa chọn (1-4): "))
            directions = ['top', 'bottom', 'left', 'right']
            direction = directions[direction_choice - 1]
            return (crop_percent, direction)
    
    elif attack_type == "Average Filter":
        param_input = input("Nhập kích thước kernel (ví dụ: 7): ")
        return int(param_input)

def main():
    input_folder = "Image512"
    watermark_path = os.path.join(input_folder, "w2_binary.png")
    
    images = [f for f in os.listdir(input_folder) if f.lower().endswith('.bmp')]
    
    attacks = [
        "Scale",
        "Salt & Pepper",
        "Gaussian Noise",
        "Blur",
        "JPEG",
        "Rotation",
        "Median",
        "Translation",
        "Histogram",
        "Lowpass Filter",
        "JPEG2000",
        "Sharpen",
        "Crop",
        "Average Filter"
    ]
    
    while True:
        clear_screen()
        print("\n=== Su New Watermark Attack System ===\n")
        
        print("Chọn chế độ tấn công:")
        print("1. Tấn công một ảnh")
        print("2. Tấn công hàng loạt ảnh")
        print("0. Thoát")
        
        try:
            mode_choice = int(input("\nNhập lựa chọn (0-2): "))
            
            if mode_choice == 0:
                break
            elif mode_choice == 1:
                # Chế độ tấn công một ảnh
                print("\nDanh sách ảnh có sẵn:")
                for i, img in enumerate(images, 1):
                    print(f"{i}. {img}")
                
                print("\nDanh sách tấn công có sẵn:")
                for i, attack in enumerate(attacks, 1):
                    print(f"{i}. {attack}")
                
                img_choice = int(input("\nChọn ảnh (số): ")) - 1
                if not 0 <= img_choice < len(images):
                    raise ValueError("Lựa chọn ảnh không hợp lệ")
                    
                attack_choice = int(input("Chọn tấn công (số): ")) - 1
                if not 0 <= attack_choice < len(attacks):
                    raise ValueError("Lựa chọn tấn công không hợp lệ")
                    
                process_attack(input_folder, watermark_path, images[img_choice], attacks[attack_choice])
                
                # Giữ nguyên kết quả và chờ người dùng nhấn phím
                print(f"\n" + "="*60)
                print("🎯 TẤN CÔNG HOÀN THÀNH!")
                print(" Kết quả đã được hiển thị ở trên")
                print("💾 File ảnh đã tấn công và watermark đã trích xuất đã được lưu")
                print("="*60)
                input("\n⏸️  Nhấn Enter để quay lại menu chính...")
                
            elif mode_choice == 2:
                # Chế độ tấn công hàng loạt - tự động chạy
                print("\nDanh sách tấn công có sẵn:")
                for i, attack in enumerate(attacks, 1):
                    print(f"{i}. {attack}")
                
                attack_choice = int(input("\nChọn tấn công (số): ")) - 1
                if not 0 <= attack_choice < len(attacks):
                    raise ValueError("Lựa chọn tấn công không hợp lệ")
                
                attack_type = attacks[attack_choice]
                attack_params = get_attack_parameters(attack_type)
                
                # Tự động chạy với tất cả ảnh
                print(f"\n🚀 Bắt đầu tấn công hàng loạt: {attack_type}")
                print(f"Tham số: {attack_params}")
                print(f"Số lượng ảnh: {len(images)}")
                print("=" * 50)
                
                # Tự động chạy với tất cả ảnh
                process_batch_attack(input_folder, watermark_path, images, attack_type, attack_params)
                
                # Giữ nguyên kết quả và chờ người dùng nhấn phím
                input("\n⏸️  Nhấn Enter để quay lại menu chính...")
                
            else:
                raise ValueError("Lựa chọn không hợp lệ")
                
        except ValueError as e:
            print(f"\nLỗi: {str(e)}")
            input("\nNhấn Enter để tiếp tục...")
        except Exception as e:
            print(f"\nLỗi không mong muốn: {str(e)}")
            input("\nNhấn Enter để tiếp tục...")

if __name__ == "__main__":
    main()