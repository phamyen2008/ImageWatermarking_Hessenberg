# Hess_new/Schur_new
import cv2
import numpy as np
import os
import time
from PIL import Image
import io

from scipy.ndimage import median_filter

from Schur_new import embed, extract, calculate_metrics # Import from su_new
from scipy.fftpack import dct, idct
from wand.image import Image

def clear_screen():
    # Lệnh xóa màn hình cho Windows
    if os.name == 'nt':
        os.system('cls')
    # Lệnh xóa màn hình cho Linux/Mac
    else:
        os.system('clear')

def scale_attack(image, scale_factor):
    h, w = image.shape[:2]
    scaled = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
    # Trả về kích thước ảnh gốc
    return cv2.resize(scaled, (w, h))

def salt_pepper_attack(image, noise_density):
    noisy_img = image.copy()
    salt_pepper_noise = np.zeros_like(image, dtype=np.uint8)
    # Tạo mảng ngẫu nhiên cho S&P
    salt_pepper_pixels = np.random.rand(image.shape[0], image.shape[1])
    
    # Tạo mask cho nhiễu Salt và Pepper
    salt_mask = salt_pepper_pixels > 1 - noise_density/2
    pepper_mask = salt_pepper_pixels < noise_density/2
    
    # Áp dụng nhiễu
    # Đặt các pixel nhiễu Pepper về 0
    noisy_img[pepper_mask] = 0
    
    # Đặt các pixel nhiễu Salt về 255
    noisy_img[salt_mask] = 255

    return noisy_img

def rotation_attack(image, angle):
    center = tuple(np.array(image.shape[0:2])/2)
    # Xoay đi
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape[0:2], flags=cv2.INTER_LINEAR)
    # Quay lại
    rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated_image = cv2.warpAffine(rotated_image, rot_mat, image.shape[:2], flags=cv2.INTER_LINEAR)
    return rotated_image


def gaussian_noise_attack(image, mean, std):
    # Chuẩn hóa về [0,1]
    img_float = image.astype(np.float32) / 255.0
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_img = np.clip(img_float + noise, 0.0, 1.0)
    return (noisy_img * 255).astype(np.uint8)


def blur_attack(image, const):
    # const là sigmaX (độ lệch chuẩn), sigmaY=0 nghĩa là bằng sigmaX
    return cv2.GaussianBlur(image, (0, 0), const)

def compress_jpeg_attack(image, quality_factor):
    temp_input = 'temp_input.bmp'
    temp_output = 'temp_output.jpg'
    
    cv2.imwrite(temp_input, image)

    attacked_img = None
    try:
        # Sử dụng wand.image để nén JPEG chất lượng cao
        with Image(filename=temp_input) as img:
            img.compression_quality = quality_factor
            img.save(filename=temp_output)

        attacked_img = cv2.imread(temp_output)
        
    except Exception as e:
        print(f"Lỗi khi sử dụng wand.image: {e}. Thử dùng cv2.imencode/imdecode thay thế.")
        # Phương án dự phòng: chỉ dùng OpenCV
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
        result, enc = cv2.imencode('.jpg', image, encode_param)
        if result:
            attacked_img = cv2.imdecode(enc, cv2.IMREAD_COLOR if image.ndim == 3 else cv2.IMREAD_GRAYSCALE)

    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(temp_output):
            os.remove(temp_output)

    if attacked_img is None:
         raise RuntimeError("Nén JPEG thất bại hoàn toàn.")
         
    return attacked_img


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
        raise ValueError("Kích thước ảnh không được hỗ trợ")


def histogram_attack(image):
    if len(image.shape) == 3:
        # Chuyển về YUV, cân bằng kênh Y, rồi chuyển lại về BGR
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return equalized
        
    else:
        # Grayscale
        return cv2.equalizeHist(image)

def lowpass_filter_attack(image, kernel_size_x, kernel_size_y):
    # Chuẩn hóa kích thước kernel về số lẻ nếu là số chẵn
    if kernel_size_x % 2 == 0:
        kernel_size_x += 1
    if kernel_size_y % 2 == 0:
        kernel_size_y += 1
    # Dùng Gaussian Blur làm Lowpass filter, sigma=0 để tính từ kernel_size
    return cv2.GaussianBlur(image, (kernel_size_x, kernel_size_y), 0)


def jp2_attack(image, compression_ratio):
    """
    Tấn công JPEG2000 với compression_ratio ∈ [1, 100]
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
    # const là sigma cho Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (0, 0), const)
    # Sharpened = (1 + alpha) * Original - alpha * Blurred 
    sharpened_image = cv2.addWeighted(image, alpha, blurred_image, beta, gamma)
    # Giới hạn giá trị
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    return sharpened_image

def crop_attack(image, crop_percent, direction=None, corner=None):
    height, width = image.shape[:2]
    img_cropped = image.copy()
    
    crop_height_px = int(height * crop_percent / 100)
    crop_width_px = int(width * crop_percent / 100)
    
    # Mảng đen để thay thế
    if image.ndim == 3:
        crop_solid = np.zeros((height, width, 3), dtype=np.uint8) # Kích thước lớn, sẽ cắt sau
    else:
        crop_solid = np.zeros((height, width), dtype=np.uint8)


    if crop_percent < 50 and corner:
        # Crop góc (thay thế 1/4 ảnh bằng màu đen)
        half_height = height // 2
        half_width = width // 2
        
        # Tạo mảng đen có kích thước của góc 1/4 
        if image.ndim == 3:
            crop_solid_corner = np.zeros((half_height, half_width, 3), dtype=np.uint8)
        else:
            crop_solid_corner = np.zeros((half_height, half_width), dtype=np.uint8)
            
        if corner == 'top-left':
            img_cropped[:half_height, :half_width] = crop_solid_corner
        elif corner == 'top-right':
            img_cropped[:half_height, half_width:] = crop_solid_corner
        elif corner == 'bottom-left':
            img_cropped[half_height:, :half_width] = crop_solid_corner
        elif corner == 'bottom-right':
            img_cropped[half_height:, half_width:] = crop_solid_corner
            
    
    elif direction:
        # Crop theo hướng (cắt một dải từ phía)
        if direction in ['left', 'right']:
            # Crop theo chiều ngang (thay đổi chiều rộng)
            if image.ndim == 3:
                crop_solid_dir = np.zeros((height, crop_width_px, 3), dtype=np.uint8)
            else:
                crop_solid_dir = np.zeros((height, crop_width_px), dtype=np.uint8)
                
            if direction == 'left':
                img_cropped[:, :crop_width_px] = crop_solid_dir
            elif direction == 'right':
                img_cropped[:, width - crop_width_px:] = crop_solid_dir
        
        elif direction in ['top', 'bottom']:
            # Crop theo chiều dọc (thay đổi chiều cao)
            if image.ndim == 3:
                crop_solid_dir = np.zeros((crop_height_px, width, 3), dtype=np.uint8)
            else:
                crop_solid_dir = np.zeros((crop_height_px, width), dtype=np.uint8)
                
            if direction == 'top':
                img_cropped[:crop_height_px, :] = crop_solid_dir
            elif direction == 'bottom':
                img_cropped[height - crop_height_px:, :] = crop_solid_dir
    
    else:
        raise ValueError("Cần chọn 'corner' (nếu < 50%) hoặc 'direction'")
        
    return img_cropped

def average_filter_attack(image, param):
    # param là kích thước kernel (ví dụ 7)
    return cv2.blur(image, (param, param))

def translation_attack(image, shift_x, shift_y):
    height, width = image.shape[:2]
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    # Tịnh tiến, phần bị dịch ra sẽ được điền bằng màu đen (borderValue=(0,0,0))
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height), borderValue=(0,0,0))
    return translated_image

def process_attack(input_folder, watermark_path, image_name, attack_type, attack_params=None):
    base_dir = "Schur_new_attacks"
    attack_folder = os.path.join(base_dir, attack_type.lower().replace(' ', '_'))
    os.makedirs(attack_folder, exist_ok=True)

    img_path = os.path.join(input_folder, image_name)
    host_image = cv2.imread(img_path)
    if host_image is None:
        print(f"Lỗi: Không thể đọc ảnh {image_name}")
        return

    print(f"\nĐang xử lý {image_name} với {attack_type}...")
    start_time = time.time()
    
    # 1. Nhúng Watermark
    watermarked_path = os.path.join(attack_folder, f"watermarked_{image_name}")
    flag_path = os.path.join(attack_folder, f"flag_{image_name}.txt")
    
    watermarked_img = embed(img_path, watermark_path, watermarked_path, flag_path)
    if watermarked_img is None:
        return

    attacked_img = None
    param_str = ""

    try:
        # 2. Thực hiện Tấn công
        if attack_type == "Scale":
            scale = attack_params
            attacked_img = scale_attack(watermarked_img, scale)
            param_str = f"_scale{scale}"
        
        elif attack_type == "Salt & Pepper":
            density = attack_params
            attacked_img = salt_pepper_attack(watermarked_img, density)
            param_str = f"_density{density}"
        
        elif attack_type == "Gaussian Noise":
            mean, std = attack_params
            attacked_img = gaussian_noise_attack(watermarked_img, mean, std)
            param_str = f"_mean{mean}_std{std}"
        
        elif attack_type == "Blur":
            const = attack_params
            attacked_img = blur_attack(watermarked_img, const)
            param_str = f"_const{const}"
        
        elif attack_type == "JPEG":
            quality_factor = attack_params
            attacked_img = compress_jpeg_attack(watermarked_img, quality_factor)
            param_str = f"_qf{quality_factor}"
        
        elif attack_type == "Rotation":
            angle = attack_params
            attacked_img = rotation_attack(watermarked_img, angle)
            param_str = f"_angle{angle}"
        
        elif attack_type == "Median":
            kernel = attack_params
            attacked_img = median_filter_attack(watermarked_img, kernel)
            param_str = f"_kernel{kernel}"

        elif attack_type == "Translation":
            shift_x, shift_y = attack_params
            attacked_img = translation_attack(watermarked_img, shift_x, shift_y)
            param_str = f"_x{shift_x}y{shift_y}"
        
        elif attack_type == "Histogram":
            attacked_img = histogram_attack(watermarked_img)
            param_str = "_hist"
        
        elif attack_type == "Lowpass Filter":
            kernel_x, kernel_y = attack_params
            attacked_img = lowpass_filter_attack(watermarked_img, kernel_x, kernel_y)
            param_str = f"_kernel{kernel_x}x{kernel_y}"
        
        elif attack_type == "JPEG2000":
            compression_ratio = attack_params
            attacked_img = jp2_attack(watermarked_img, compression_ratio)
            param_str = f"_cr{compression_ratio}"
        
        elif attack_type == "Sharpen":
            const = attack_params
            attacked_img = sharpen_attack(watermarked_img, const)
            param_str = f"_const{const}"
        
        elif attack_type == "Crop":
            crop_percent, param_val = attack_params
            if isinstance(param_val, str) and param_val in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
                 # Corner crop
                 attacked_img = crop_attack(watermarked_img, crop_percent, corner=param_val)
                 param_str = f"_crop{crop_percent}_{param_val.replace('-', '_')}"
            elif isinstance(param_val, str) and param_val in ['top', 'bottom', 'left', 'right']:
                 # Direction crop
                 attacked_img = crop_attack(watermarked_img, crop_percent, direction=param_val)
                 param_str = f"_crop{crop_percent}_{param_val}"
            else:
                 raise ValueError("Tham số Crop không hợp lệ")
        
        elif attack_type == "Average Filter":
            param = attack_params
            attacked_img = average_filter_attack(watermarked_img, param)
            param_str = f"_avg{param}"
        
        else:
            raise ValueError(f"Loại tấn công '{attack_type}' không được hỗ trợ.")
            
        
        # 3. Lưu ảnh bị tấn công
        attack_name = attack_type.lower().replace(' ', '_').replace('&', 'and')
        attacked_folder = os.path.join(attack_folder, f"{attack_name}_images")
        extracted_folder = os.path.join(attack_folder, "extracted_watermarks")
        os.makedirs(attacked_folder, exist_ok=True)
        os.makedirs(extracted_folder, exist_ok=True)

        base_name = os.path.splitext(image_name)[0]
        attacked_path = os.path.join(attacked_folder, f"{attack_name}_{base_name}{param_str}.bmp")
        cv2.imwrite(attacked_path, attacked_img)

        # 4. Trích xuất Watermark và Tính Metrics
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        
        # Trích xuất từ ảnh đã nhúng gốc (để có NC_orig)
        original_extracted = extract(watermarked_path, flag_path,
                                 os.path.join(extracted_folder, f"original_extracted_{base_name}.png"))
                                 
        # Trích xuất từ ảnh đã bị tấn công
        extracted_watermark = extract(attacked_path, flag_path,
                                      os.path.join(extracted_folder, f"extracted_{base_name}{param_str}.png"))

        if original_extracted is not None and extracted_watermark is not None and watermark is not None:
            # Metrics cho ảnh đã nhúng (độ vô hình và NC gốc)
            psnr_orig, ssim_orig, nc_orig = calculate_metrics(
                host_image, watermarked_img, original_extracted, watermark
            )
            # Metrics cho ảnh đã bị tấn công (độ bền)
            psnr_val, ssim_val, nc_val = calculate_metrics(
                host_image, attacked_img, extracted_watermark, watermark
            )

            # 5. In Kết quả
            print(f"\n{image_name} - Original Metrics (Ảnh đã nhúng/Watermark gốc):")
            print("---------------------------")
            print(f"PSNR (Host/WM): {psnr_orig:.2f}")
            print(f"SSIM (Host/WM): {ssim_orig:.4f}")
            print(f"NC (WM/Extracted_WM): {nc_orig:.4f}")

            print(f"\n{image_name} - {attack_type} Results (Ảnh đã tấn công/Watermark trích xuất):")
            print("---------------------------")
            print(f"PSNR (Host/Attacked): {psnr_val:.2f}")
            print(f"SSIM (Host/Attacked): {ssim_val:.4f}")
            print(f"NC (WM/Extracted_Attacked): {nc_val:.4f}")
            print(f"Suy giảm NC: {nc_orig - nc_val:.4f}")

        total_time = time.time() - start_time
        print(f"\nTổng thời gian xử lý: {total_time:.4f} giây")
        
        return {
            'image_name': image_name,
            'attack_type': attack_type,
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
        print(f"\nLỗi: Đầu vào không hợp lệ - {str(e)}")
        return None
    except Exception as e:
        print(f"\nLỗi không mong muốn: {str(e)}")
        # Dọn dẹp file tạm
        if os.path.exists('temp_input.bmp'): os.remove('temp_input.bmp')
        if os.path.exists('temp_output.jpg'): os.remove('temp_output.jpg')
        return None


def get_attack_parameters(attack_type):
    """Lấy tham số tấn công từ người dùng"""
    if attack_type == "Scale":
        scale_input = input("Nhập hệ số scale (ví dụ: 0.5, 2.0): ")
        return float(scale_input)
    
    elif attack_type == "Salt & Pepper":
        density_input = input("Nhập mật độ nhiễu (0-1): ")
        density = float(density_input)
        if not (0 <= density <= 1):
             raise ValueError("Mật độ nhiễu phải nằm trong khoảng [0, 1].")
        return density
    
    elif attack_type == "Gaussian Noise":
        mean_input = input("Nhập mean (ví dụ: 0): ")
        std_input = input("Nhập standard deviation (ví dụ: 0.05): ")
        return (float(mean_input), float(std_input))
    
    elif attack_type == "Blur":
        const_input = input("Nhập hằng số blur (sigmaX, ví dụ: 0.4): ")
        return float(const_input)
    
    elif attack_type == "JPEG":
        print("\nChọn quality factor:")
        print("1. QF = 50")
        print("2. QF = 70")
        print("3. QF = 90")
        qf_choice = int(input("Nhập lựa chọn (1-3): "))
        quality_factors = [50, 70, 90]
        if not (1 <= qf_choice <= 3):
             raise ValueError("Lựa chọn QF không hợp lệ.")
        return quality_factors[qf_choice - 1]
    
    elif attack_type == "Rotation":
        angle_input = input("Nhập góc xoay (độ, ví dụ: 1): ")
        return float(angle_input)
    
    elif attack_type == "Median":
        kernel_input = input("Nhập kích thước kernel (số lẻ, ví dụ: 3): ")
        kernel = int(kernel_input)
        if kernel % 2 == 0:
             raise ValueError("Kích thước kernel Median phải là số lẻ.")
        return kernel
    
    elif attack_type == "Translation":
        shift_x_input = input("Nhập dịch chuyển ngang (pixels, ví dụ: 5): ")
        shift_y_input = input("Nhập dịch chuyển dọc (pixels, ví dụ: -5): ")
        return (int(shift_x_input), int(shift_y_input))
    
    elif attack_type == "Histogram":
        return None
    
    elif attack_type == "Lowpass Filter":
        print("Nhập kích thước kernel (số lẻ, ví dụ: 3 3 hoặc 100 1):")
        kernel_x = int(input("Kernel size X: "))
        kernel_y = int(input("Kernel size Y: "))
        return (kernel_x, kernel_y)
    
    elif attack_type == "JPEG2000":
        compression_ratio_input = input("Nhập tỷ lệ nén (1-100, 1=ít nén, 100=nén nhiều): ")
        compression_ratio = float(compression_ratio_input)
        if not (1 <= compression_ratio <= 100):
             raise ValueError("Tỷ lệ nén JPEG2000 phải nằm trong khoảng [1, 100].")
        return compression_ratio
    
    elif attack_type == "Sharpen":
        const_input = input("Nhập hằng số sharpening (sigma cho Blur, khuyến nghị > 1, ví dụ: 2): ")
        return float(const_input)
    
    elif attack_type == "Crop":
        crop_percent_input = input("Nhập phần trăm crop (0-100): ")
        crop_percent = float(crop_percent_input)
        if not (0 <= crop_percent <= 100):
             raise ValueError("Phần trăm Crop phải nằm trong khoảng [0, 100].")
             
        if crop_percent < 50:
            print("\nChọn loại crop:")
            print("1. Corner cropping (Cắt một góc 1/4 ảnh)")
            print("2. Direction cropping (Cắt một dải từ phía)")
            crop_type = int(input("Nhập lựa chọn (1-2): "))
            
            if crop_type == 1:
                print("\nChọn góc để crop:")
                print("1. Top-Left")
                print("2. Top-Right")
                print("3. Bottom-Left")
                print("4. Bottom-Right")
                corner_choice = int(input("Nhập lựa chọn (1-4): "))
                corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
                if not (1 <= corner_choice <= 4):
                     raise ValueError("Lựa chọn góc crop không hợp lệ.")
                corner = corners[corner_choice - 1]
                return (crop_percent, corner)
            else:
                print("\nChọn hướng crop (Direction cropping):")
                print("1. Top")
                print("2. Bottom")
                print("3. Left")
                print("4. Right")
                direction_choice = int(input("Nhập lựa chọn (1-4): "))
                directions = ['top', 'bottom', 'left', 'right']
                if not (1 <= direction_choice <= 4):
                     raise ValueError("Lựa chọn hướng crop không hợp lệ.")
                direction = directions[direction_choice - 1]
                return (crop_percent, direction)
        else:
            # Chỉ cho phép Direction crop nếu crop_percent >= 50
            print("\nChọn hướng crop (Direction cropping):")
            print("1. Top")
            print("2. Bottom")
            print("3. Left")
            print("4. Right")
            direction_choice = int(input("Nhập lựa chọn (1-4): "))
            directions = ['top', 'bottom', 'left', 'right']
            if not (1 <= direction_choice <= 4):
                 raise ValueError("Lựa chọn hướng crop không hợp lệ.")
            direction = directions[direction_choice - 1]
            return (crop_percent, direction)
    
    elif attack_type == "Average Filter":
        param_input = input("Nhập kích thước kernel (số lẻ, ví dụ: 7): ")
        param = int(param_input)
        if param % 2 == 0:
             raise ValueError("Kích thước kernel Average Filter nên là số lẻ.")
        return param
        
    raise ValueError(f"Tham số cho {attack_type} không xác định.")


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
        print("\n=== Su New Watermark Attack System (Chế độ một ảnh) ===\n")
        
        # 1. Chọn ảnh
        print("Danh sách ảnh có sẵn (.bmp):")
        if not images:
            print(f"Lỗi: Không tìm thấy ảnh .bmp nào trong thư mục '{input_folder}'")
            input("\nNhấn Enter để thoát...")
            break
            
        for i, img in enumerate(images, 1):
            print(f"{i}. {img}")
        print("0. Thoát")
        
        try:
            img_choice_input = input("\nChọn ảnh (số 1-{len(images)}): ")
            if img_choice_input == '0':
                break
                
            img_choice = int(img_choice_input) - 1
            if not 0 <= img_choice < len(images):
                raise ValueError("Lựa chọn ảnh không hợp lệ")
            
            selected_image = images[img_choice]
            
            # 2. Chọn loại tấn công
            print("\n" + "="*50)
            print("Danh sách tấn công có sẵn:")
            for i, attack in enumerate(attacks, 1):
                print(f"{i}. {attack}")
            
            attack_choice = int(input("Chọn tấn công (số): ")) - 1
            if not 0 <= attack_choice < len(attacks):
                raise ValueError("Lựa chọn tấn công không hợp lệ")
            
            attack_type = attacks[attack_choice]
            
            # 3. Lấy tham số
            print("\n" + "="*50)
            print(f"Nhập tham số cho tấn công: {attack_type}")
            attack_params = get_attack_parameters(attack_type)
            
            # 4. Thực hiện tấn công và hiển thị kết quả
            print("\n" + "="*50)
            process_attack(input_folder, watermark_path, selected_image, attack_type, attack_params)
            
            # 5. Giữ màn hình
            print(f"\n" + "="*60)
            print("🎯 TẤN CÔNG HOÀN THÀNH!")
            print(" Kết quả đã được hiển thị ở trên.")
            print("💾 File ảnh đã tấn công và watermark đã trích xuất đã được lưu trong thư mục Schur_new_attacks/")
            print("="*60)
            input("\n⏸️  Nhấn Enter để quay lại menu chính...")
            
        except ValueError as e:
            print(f"\nLỗi: {str(e)}")
            input("\nNhấn Enter để tiếp tục...")
        except Exception as e:
            print(f"\nLỗi không mong muốn: {str(e)}")
            input("\nNhấn Enter để tiếp tục...")

if __name__ == "__main__":
    main()
