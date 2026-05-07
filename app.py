# app.py
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0"
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import io
import torch
import torch.nn.functional as F
import torch.quantization # <<< THÊM THƯ VIỆN LƯỢNG TỬ HÓA
import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import base64
from google import genai
from google.genai import types
import random
from collections import Counter
import markdown
import re
import json

from werkzeug.utils import secure_filename
from dotenv import load_dotenv
# --- THÊM CÁC THƯ VIỆN CHO GRAD-CAM ---
import cv2 
# THÊM CẢ 3 LOẠI: LayerCAM (mặc định), ScoreCAM (chi tiết)
from pytorch_grad_cam import LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- HÀM RESHAPE CHO ViT (PVT) ---
def reshape_transform(tensor):
    # Input tensor shape: (B, N, C), where N = H*W
    # (pvt_v2_b5, 224x224, final stage) -> N=49 (7x7)
    B, N, C = tensor.shape
    H = W = int(N**0.5)
    # Giả định N là một số chính phương (ví dụ: 49)
    # Reshape (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
    return tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)

class CFG:
    seed = 42
    device = torch.device("cpu") # Mặc định an toàn là CPU
    if torch.cuda.is_available():
        try:
            # Thử chạy một phép tính nhỏ trên GPU để bắt lỗi "no kernel image" sớm
            _ = torch.tensor([1.0]).to("cuda")
            device = torch.device("cuda")
        except Exception as e:
            print(f"⚠️ Cảnh báo: Tìm thấy GPU nhưng xảy ra lỗi ({e}). Đang tự động chuyển sang sử dụng CPU...")
    img_size = 224
    model_dir = "models"
    model_name = 'pvt_v2_b5.in1k'

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
try:
    client = genai.Client(api_key=API_KEY)
    print("✅ API Gemini đã được cấu hình thành công.")
    gemini_ready = True
except Exception as e:
    print(f"❌ LỖI: Không thể cấu hình API Gemini: {e}")
    client = None
    gemini_ready = False

SYSTEM_PROMPT = """
Bạn là một trợ lý ảo tên là 'EyeCare', hoạt động trên một trang web sàng lọc bệnh về mắt bằng AI.
Nhiệm vụ của bạn là:
1.  Trả lời các câu hỏi chung về các bệnh mắt như võng mạc tiểu đường, đục thủy tinh thể, glôcôm.
2.  Hướng dẫn người dùng cách sử dụng trang web.
3.  Nếu người dùng cung cấp hình ảnh và mô tả triệu chứng, bạn có thể đưa ra nhận định sơ bộ, nhưng BẮT BUỘC phải kèm theo lời khuyên đi khám bác sĩ.
4.  Luôn luôn giữ thái độ đồng cảm, lịch sự và sử dụng ngôn ngữ dễ hiểu.
5.  Nếu phát hiện kết quả nhiều phần không chắc chắn, hãy hiển thị cảnh báo hoặc gợi ý người dùng xác minh.
6.  Kiểm tra độ tin cậy của nội dung AI tạo ra bằng cách đối chiếu với nguồn dữ liệu đáng tin cậy (web đăng tải bài báo nghiên cứu khoa học, cơ sở dữ liệu nội bộ, báo uy tín).
**QUY TẮC BẤT DI BẤT DỊCH:**
-   **TUYỆT ĐỐI KHÔNG ĐƯỢC PHÉP ĐƯA RA CHẨN ĐOÁN Y KHOA CUỐI CÙNG.** Luôn kết thúc các câu trả lời liên quan đến bệnh lý bằng câu: "Thông tin này chỉ mang tính tham khảo. Để có kết quả chính xác nhất, bạn vui lòng đến gặp bác sĩ chuyên khoa mắt."
-   Không được đưa ra lời khuyên về liều lượng thuốc men.
-   Nếu được hỏi những câu không liên quan đến sức khỏe mắt hoặc y tế, hãy lịch sự từ chối.
-   Tuyệt đối không được yêu cầu người dùng đăng ký tài khoản
"""

FACT_CHECK_PROMPT_TEMPLATE = """
You are a fact-checking expert. Verify the accuracy of the following claim using reliable, searchable sources.

INPUT CLAIM: "{fact}"

TASK: Search for credible sources to verify the claim's accuracy and return ONLY a JSON object in the specified format. Do not include any text before or after the JSON object.

OUTPUT JSON FORMAT:
{{
  "claim_analysis": {{
    "original_claim": "[exact claim being verified]",
    "verification_status": "[TRUE/FALSE/PARTIALLY_TRUE/UNVERIFIED]",
    "confidence_level": "[HIGH/MEDIUM/LOW]",
    "summary": "[brief explanation of findings, max 100 words, in Vietnamese]"
  }},
  "evidence": [
    {{
      "source_title": "[title of source]",
      "source_url": "[direct link to source]",
      "source_credibility": "[HIGH/MEDIUM/LOW]",
      "relevant_excerpt": "[key information from source, max 50 words, in Vietnamese]",
      "supports_claim": "[SUPPORTS/CONTRADICTS/NEUTRAL]"
    }}
  ],
  "conclusion": {{
    "final_verdict": "[TRUE/FALSE/PARTIALLY_TRUE/UNVERIFIED]",
    "explanation": "[detailed reasoning, max 150 words, in Vietnamese]",
    "recommendation": "[ACCEPT/REJECT/INVESTIGATE_FURTHER]"
  }}
}}
"""


# --- MODEL & TRANSFORMS ---
def get_transforms(is_train=False):
    return A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class OcularModel(torch.nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.model.num_features
        self.model.head = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

@torch.no_grad()
def get_prediction_and_probabilities(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.to(CFG.device).unsqueeze(0)
    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
    pred_index = np.argmax(probabilities)
    confidence = probabilities[pred_index]
    return pred_index, confidence, probabilities

def run_full_prediction_pipeline_with_details(image_tensor, models):
    _, _, probs_s1 = get_prediction_and_probabilities(models['s1'], image_tensor)
    p_cataract = probs_s1[0]
    p_not_cataract = probs_s1[1]

    _, _, probs_s2 = get_prediction_and_probabilities(models['s2'], image_tensor)
    p_normal = probs_s2[0]
    p_not_normal = probs_s2[1]

    _, _, probs_s3 = get_prediction_and_probabilities(models['s3'], image_tensor)
    p_dr = probs_s3[0]
    p_glaucoma = probs_s3[1]

    final_probs = {
        "Đục thủy tinh thể": p_cataract,
        "Bình thường": p_not_cataract * p_normal,
        "Bệnh võng mạc tiểu đường": p_not_cataract * p_not_normal * p_dr,
        "Glôcôm": p_not_cataract * p_not_normal * p_glaucoma
    }

    winning_label = max(final_probs, key=final_probs.get)
    winning_confidence = final_probs[winning_label]

    return winning_label, winning_confidence, final_probs

def get_all_filters():
    return {
        'brightness_contrast': A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
        'gamma': A.RandomGamma(gamma_limit=(80, 120), p=1),
        'clahe': A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
        'hue_saturation': A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
        'color_jitter': A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
        'rgb_shift': A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1),
        'sharpen': A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
        'unsharp_mask': A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.5, alpha=(0.2, 0.5), threshold=10, p=1),
        'emboss': A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1),
        # --- SỬA CẢNH BÁO ---
        # Đã đổi 'var_limit' thành 'variance_limit' để sửa UserWarning
        'gauss_noise': A.GaussNoise(variance_limit=(10.0, 50.0), p=1),
        'iso_noise': A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
        'blur': A.GaussianBlur(blur_limit=(3, 7), p=1),
        'channel_shuffle': A.ChannelShuffle(p=1)
    }

# --- HÀM HỖ TRỢ CHUYỂN ẢNH PIL SANG BASE64 ---
def pil_to_base64(pil_img, format="JPEG"):
    """Chuyển đổi đối tượng PIL Image sang chuỗi Base64."""
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

# --- HÀM HỖ TRỢ TÁI CẤU TRÚC (REFACTOR) ---
def get_explanation_targets(winning_label, all_probs, models):
    """
    Dựa trên kết quả chẩn đoán, trả về model và class_index
    để dùng cho Grad-CAM.
    LƯU Ý: Hàm này PHẢI nhận dict model FP32 (chưa lượng tử hóa).
    """
    model_to_explain = None
    target_class_index = None

    if winning_label == "Đục thủy tinh thể":
        model_to_explain = models['s1']
        target_class_index = 0 # Class "Đục thủy tinh thể" trong model s1
    elif all_probs["Đục thủy tinh thể"] < 0.5: # Nếu không phải đục thủy tinh thể
        if winning_label == "Bình thường":
            model_to_explain = models['s2']
            target_class_index = 0 # Class "Bình thường" trong model s2
        elif winning_label == "Bệnh võng mạc tiểu đường":
            model_to_explain = models['s3']
            target_class_index = 0 # Class "Bệnh võng mạc tiểu đường" trong model s3
        elif winning_label == "Glôcôm":
            model_to_explain = models['s3']
            target_class_index = 1 # Class "Glôcôm" trong model s3
    
    return model_to_explain, target_class_index

# --- FLASK APPLICATION ---
app = Flask(__name__)

# --- LOAD AI MODELS ---
# SỬA LỖI: Tách model cho inference (quantized) và model cho Grad-CAM (fp32)
models = {} # Model cho inference (sẽ được lượng tử hóa nếu là CPU)
fp32_models_for_gradcam = {} # Model FP32 gốc cho Grad-CAM
models_loaded_successfully = False
try:
    print("--- Bắt đầu tải các model AI ---")
    
    # --- Model S1 ---
    model_s1_path = os.path.join(CFG.model_dir, 'model_stage1.pth')
    model_s1_fp32 = OcularModel(CFG.model_name, num_classes=2, pretrained=False).to(CFG.device)
    model_s1_fp32.load_state_dict(torch.load(model_s1_path, map_location=CFG.device))
    model_s1_fp32.eval()
    fp32_models_for_gradcam['s1'] = model_s1_fp32 # Lưu model FP32 cho Grad-CAM
    
    # --- Model S2 ---
    model_s2_path = os.path.join(CFG.model_dir, 'model_stage2.pth')
    model_s2_fp32 = OcularModel(CFG.model_name, num_classes=2, pretrained=False).to(CFG.device)
    model_s2_fp32.load_state_dict(torch.load(model_s2_path, map_location=CFG.device))
    model_s2_fp32.eval()
    fp32_models_for_gradcam['s2'] = model_s2_fp32 # Lưu model FP32 cho Grad-CAM
    
    # --- Model S3 ---
    model_s3_path = os.path.join(CFG.model_dir, 'model_stage3.pth')
    model_s3_fp32 = OcularModel(CFG.model_name, num_classes=2, pretrained=False).to(CFG.device)
    model_s3_fp32.load_state_dict(torch.load(model_s3_path, map_location=CFG.device))
    model_s3_fp32.eval()
    fp32_models_for_gradcam['s3'] = model_s3_fp32 # Lưu model FP32 cho Grad-CAM

    # --- TỐI ƯU HÓA: LƯỢNG TỬ HÓA (QUANTIZATION) NẾU CHẠY TRÊN CPU ---
    if CFG.device.type == 'cpu':
        print("--- Đang áp dụng lượng tử hóa INT8 cho CPU... ---")
        # Lượng tử hóa giúp CPU chạy nhanh gần bằng GPU
        models['s1'] = torch.quantization.quantize_dynamic(
            model_s1_fp32, {torch.nn.Linear}, dtype=torch.qint8
        )
        models['s2'] = torch.quantization.quantize_dynamic(
            model_s2_fp32, {torch.nn.Linear}, dtype=torch.qint8
        )
        models['s3'] = torch.quantization.quantize_dynamic(
            model_s3_fp32, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("✅ Lượng tử hóa INT8 hoàn tất.")
    else:
        print("--- Đang chạy trên CUDA (GPU). ---")
        models['s1'] = model_s1_fp32
        models['s2'] = model_s2_fp32
        models['s3'] = model_s3_fp32

    models_loaded_successfully = True
    print("✅ Tất cả model AI đã được tải và cấu hình thành công.")
except Exception as e:
    print(f"❌ LỖI: Không thể tải hoặc lượng tử hóa model AI: {e}")
    models.clear()
    fp32_models_for_gradcam.clear()
    models_loaded_successfully = False

# --- WEB ROUTES ---
@app.route('/')
def serve_index():
    # CẬP NHẬT: Dùng render_template của Flask
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image_route():
    if not models_loaded_successfully:
        return jsonify({'error': 'Hệ thống AI chưa sẵn sàng.'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'Không có file ảnh được cung cấp.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn.'}), 400
    
    # Lấy tham số 'mode' từ request. Mặc định là 'deep'
    mode = request.form.get('mode', 'deep')
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # --- TẠO ẢNH BASE64 ---
        # Giữ một bản sao base64 của ảnh gốc (chưa resize) để hiển thị
        original_image_b64 = pil_to_base64(image)
        
        image_np = np.array(image)
        base_transforms = get_transforms()
        
        original_tensor = base_transforms(image=image_np)['image']
        
        # --- CHẠY DỰ ĐOÁN GỐC ĐỂ XÁC ĐỊNH HEATMAP ---
        # Dùng model đã tối ưu (quantized) để dự đoán nhanh
        label, conf, all_probs = run_full_prediction_pipeline_with_details(original_tensor, models)
        winning_label = label
        
        # --- TẠO GRAD-CAM (MẶC ĐỊNH DÙNG LAYER CAM) ---
        heatmap_image_b64 = None
        
        # SỬA LỖI: Dùng model FP32 (chưa lượng tử hóa) cho Grad-CAM
        model_to_explain, target_class_index = get_explanation_targets(winning_label, all_probs, fp32_models_for_gradcam)
        
        if model_to_explain:
            try:
                # 1. Chuẩn bị ảnh (resize, chưa normalize) cho việc chồng heatmap
                image_resized_pil = image.resize((CFG.img_size, CFG.img_size))
                image_for_overlay_np = np.array(image_resized_pil)
                image_for_overlay_float = np.float32(image_for_overlay_np) / 255
                
                # 2. Xác định lớp mục tiêu
                target_layer = model_to_explain.model.stages[-1].blocks[-1].norm2
                
                # 3. Chuẩn bị tensor đầu vào cho Grad-CAM (cần batch, device và gradient)
                # Model FP32 vẫn cần tensor trên đúng device
                gradcam_input_tensor = original_tensor.to(CFG.device).unsqueeze(0).requires_grad_(True)
                
                # 4. Khởi tạo GradCam (Mặc định dùng LayerCAM)
                cam = LayerCAM(model=model_to_explain, 
                             target_layers=[target_layer],
                             reshape_transform=reshape_transform)
                targets = [ClassifierOutputTarget(target_class_index)]
                
                # 5. Chạy Grad-CAM
                grayscale_cam = cam(input_tensor=gradcam_input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :] # Lấy heatmap đầu tiên
                
                # 6. Tạo ảnh heatmap (cam_image)
                cam_image = show_cam_on_image(image_for_overlay_float, grayscale_cam, use_rgb=True, image_weight=0.4)
                
                # 7. *** TẠO MASK ĐỂ BỎ NỀN ĐEN ***
                gray_img = cv2.cvtColor(image_for_overlay_np, cv2.COLOR_RGB2GRAY)
                background_mask = (gray_img < 20).astype(np.uint8) # 1 = background, 0 = retina
                background_mask_3d = cv2.cvtColor(background_mask * 255, cv2.COLOR_GRAY2RGB)
                cam_image_masked = np.where(background_mask_3d == 255, 0, cam_image)
                
                # 8. Chuyển sang PIL và base64
                cam_pil = Image.fromarray(cam_image_masked.astype(np.uint8))
                heatmap_image_b64 = pil_to_base64(cam_pil)
                print("✅ LayerCAM heatmap (mặc định) đã được tạo thành công (với mask).")
                
            except Exception as e:
                print(f"❌ LỖI khi tạo Grad-CAM (LayerCAM): {e}. Bỏ qua heatmap.")
                heatmap_image_b64 = None
        
        # --- KẾT THÚC TẠO GRAD-CAM ---

        if mode == 'fast':
            # Chẩn đoán nhanh, chỉ 1 lần dự đoán
            
            # Trả về kết quả cho chẩn đoán nhanh
            final_prediction_label = label
            vote_percentage = conf * 100
            
            color_code = 'red'
            if vote_percentage > 90: color_code = 'green'
            elif vote_percentage > 75: color_code = 'yellow'
            elif vote_percentage > 50: color_code = 'orange'

            return jsonify({
                'prediction': final_prediction_label,
                'vote_percentage': float(vote_percentage),
                'color': color_code,
                'chart_data': [], # Không có dữ liệu biểu đồ
                'vote_distribution': [], # Không có phân phối vote
                'original_image_b64': original_image_b64,
                'heatmap_image_b64': heatmap_image_b64
            })
            
        elif mode == 'deep':
            # Chẩn đoán sâu (mặc định)
            all_run_details = []

            # Thêm kết quả của ảnh gốc vào (đã chạy ở trên)
            all_run_details.append({
                'run': 1, 'label': label, 'confidence': conf, 'all_probs': all_probs
            })
            
            NUM_VOTES = 50
            all_filters = get_all_filters()
            filter_keys = list(all_filters.keys())
            for i in range(NUM_VOTES):
                random.seed(i)
                np.random.seed(i)
                chosen_keys = random.sample(filter_keys, 5)
                chosen_filters = [all_filters[key] for key in chosen_keys]
                augment_pipeline = A.Compose(chosen_filters)
                augmented_image_np = augment_pipeline(image=image_np)['image']
                augmented_tensor = base_transforms(image=augmented_image_np)['image']
                
                # Dùng model đã tối ưu (quantized) cho 50 lần lặp
                label_aug, conf_aug, all_probs_aug = run_full_prediction_pipeline_with_details(augmented_tensor, models)
                all_run_details.append({
                    'run': i + 2, 'label': label_aug, 'confidence': conf_aug, 'all_probs': all_probs_aug
                })
            
            all_votes = [detail['label'] for detail in all_run_details]
            vote_counts = Counter(all_votes)
            
            vote_distribution = []
            for label_item, count in vote_counts.items():
                percentage = (count / len(all_votes)) * 100
                vote_distribution.append({'label': label_item, 'percentage': float(percentage)})
            
            vote_distribution.sort(key=lambda x: x['percentage'], reverse=True)
            
            final_prediction_label = vote_distribution[0]['label']
            vote_percentage = vote_distribution[0]['percentage']

            chart_data = []
            for detail in all_run_details:
                is_vote_true = detail['label'] == final_prediction_label
                y_value = detail['confidence'] if is_vote_true else detail['all_probs'].get(final_prediction_label, 0.0)
                
                chart_data.append({
                    'x': detail['run'],
                    'y': float(y_value) * 100,
                    'vote_true': is_vote_true
                })
            
            color_code = 'red'
            if vote_percentage > 90: color_code = 'green'
            elif vote_percentage > 75: color_code = 'yellow'
            elif vote_percentage > 50: color_code = 'orange'
            
            return jsonify({
                'prediction': final_prediction_label,
                'vote_percentage': float(vote_percentage),
                'color': color_code,
                'chart_data': chart_data,
                'vote_distribution': vote_distribution,
                'original_image_b64': original_image_b64,
                'heatmap_image_b64': heatmap_image_b64
            })
        
    except Exception as e:
        print(f"Lỗi dự đoán: {e}")
        return jsonify({'error': f'Lỗi khi xử lý ảnh hoặc dự đoán: {str(e)}'}), 500


# --- ROUTE MỚI ĐỂ TẠO HEATMAP CHI TIẾT (SCORECAM) ---
@app.route('/generate-heatmap', methods=['POST'])
def generate_heatmap_route():
    if not models_loaded_successfully:
        return jsonify({'error': 'Hệ thống AI chưa sẵn sàng.'}), 503
    
    try:
        data = request.get_json()
        image_b64 = data.get('image_b64')
        prediction_label = data.get('prediction_label')
        heatmap_method = data.get('heatmap_method', 'ScoreCAM') # Mặc định là ScoreCAM cho route này

        if not image_b64 or not prediction_label:
            return jsonify({'error': 'Thiếu ảnh hoặc nhãn dự đoán.'}), 400

        # 1. Giải mã ảnh Base64
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # 2. Lấy tensor đã chuẩn hóa
        base_transforms = get_transforms()
        input_tensor = base_transforms(image=image_np)['image']
        
        # 3. Chạy lại dự đoán (nhanh) để lấy all_probs (dùng model tối ưu)
        _, _, all_probs = run_full_prediction_pipeline_with_details(input_tensor, models)

        # 4. Lấy model và target class
        # SỬA LỖI: Dùng model FP32 (chưa lượng tử hóa) cho Grad-CAM
        model_to_explain, target_class_index = get_explanation_targets(prediction_label, all_probs, fp32_models_for_gradcam)
        
        if not model_to_explain:
            return jsonify({'error': 'Không thể xác định model để giải thích.'}), 500

        # 5. Chuẩn bị ảnh nền
        image_resized_pil = image.resize((CFG.img_size, CFG.img_size))
        image_for_overlay_np = np.array(image_resized_pil)
        image_for_overlay_float = np.float32(image_for_overlay_np) / 255
        
        # 6. Xác định lớp mục tiêu
        target_layer = model_to_explain.model.stages[-1].blocks[-1].norm2
        
        # 7. Chuẩn bị tensor đầu vào
        # Model FP32 vẫn cần tensor trên đúng device
        cam_input_tensor = input_tensor.to(CFG.device).unsqueeze(0)
        
        # 8. Khởi tạo và chạy ScoreCAM
        print(f"--- Bắt đầu tạo {heatmap_method} (có thể chậm) ---")
        cam = ScoreCAM(model=model_to_explain, 
                       target_layers=[target_layer],
                       reshape_transform=reshape_transform)
        
        targets = [ClassifierOutputTarget(target_class_index)]
        
        grayscale_cam = cam(input_tensor=cam_input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # 9. Tạo ảnh heatmap (cam_image)
        cam_image = show_cam_on_image(image_for_overlay_float, grayscale_cam, use_rgb=True, image_weight=0.4)
        
        # 10. Áp dụng Mask
        gray_img = cv2.cvtColor(image_for_overlay_np, cv2.COLOR_RGB2GRAY)
        background_mask = (gray_img < 20).astype(np.uint8) # 1 = background, 0 = retina
        background_mask_3d = cv2.cvtColor(background_mask * 255, cv2.COLOR_GRAY2RGB)
        cam_image_masked = np.where(background_mask_3d == 255, 0, cam_image)
        
        # 11. Chuyển sang PIL và base64
        cam_pil = Image.fromarray(cam_image_masked.astype(np.uint8))
        heatmap_image_b64 = pil_to_base64(cam_pil)
        print(f"✅ {heatmap_method} heatmap đã được tạo thành công.")
        
        return jsonify({ 'heatmap_image_b64': heatmap_image_b64 })

    except Exception as e:
        print(f"❌ LỖI khi tạo Grad-CAM (ScoreCAM): {e}.")
        return jsonify({'error': f'Lỗi khi tạo heatmap chi tiết: {str(e)}'}), 500


# --- CÁC ENDPOINT CHO CHATBOT VÀ KIỂM CHỨNG ---

@app.route('/fact-check', methods=['POST'])
def fact_check():
    if not gemini_ready:
        return jsonify({'error': 'Dịch vụ AI hiện không khả dụng.'}), 503
    try:
        data = request.get_json()
        fact_to_check = data.get('fact', '')
        if not fact_to_check:
            return jsonify({'error': 'Không có nội dung để kiểm tra'}), 400

        prompt = FACT_CHECK_PROMPT_TEMPLATE.format(fact=fact_to_check)
        
        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        # --- SỬA LỖI 404 ---
        # Đã đổi "gemini-1.5-flash" thành "gemini-1.5-flash-latest"
        response = client.models.generate_content(
            model='gemini-flash-latest',
            contents=prompt,
            config=generation_config,
        )
        
        analysis_result = json.loads(response.text)
        
        return jsonify({'success': True, 'result': analysis_result})

    except json.JSONDecodeError as e:
        print(f"Lỗi giải mã JSON từ phản hồi của AI: {e}")
        print(f"Phản hồi nhận được: {response.text}")
        return jsonify({'error': 'Lỗi giải mã phản hồi JSON từ AI.'}), 500
    except Exception as e:
        print(f"Lỗi tại endpoint /fact-check: {e}")
        return jsonify({'error': f'Có lỗi xảy ra phía server: {str(e)}'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    if not gemini_ready:
        return jsonify({'error': 'Dịch vụ chatbot hiện không khả dụng.'}), 503
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '')
        # image_b64 = data.get('image_data') # Đã xóa logic đính kèm ảnh
        if not user_prompt: # Chỉ kiểm tra user_prompt
            return jsonify({'error': 'Không có nội dung đầu vào'}), 400
        
        prompt_parts = [SYSTEM_PROMPT, user_prompt]
        
        # Đã xóa logic xử lý image_b64
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_parts
        )
        
        html_response = markdown.markdown(response.text)

        return jsonify({'response': html_response})

    except Exception as e:
        print(f"Lỗi tại endpoint /chat: {e}")
        return jsonify({'error': f'Có lỗi xảy ra phía server: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5150)