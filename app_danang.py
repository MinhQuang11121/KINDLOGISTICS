import streamlit as st
import pandas as pd
import numpy as np
import torch
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from model import STGCN 


# ==========================================
# 0. KHỞI TẠO BỘ NÃO AI (CACHE ĐỂ TỐI ƯU BỘ NHỚ)
# ==========================================
@st.cache_resource
def load_ai_system():
    device = torch.device("cpu") 
    num_nodes = 2891       
    in_features = 6        
    hidden_features = 128  
    num_layers = 3 
    
    try:
        model = STGCN(num_nodes, in_features, hidden_features, 12, 3, num_layers).to(device)
        model.load_state_dict(torch.load('models_weights/danang_ultra_v6.pth', map_location=device))
        model.eval() 
        # Giả sử đoạn code load model của ông dạng như này:
# model = STGCN(...)
# model.load_state_dict(torch.load('models_weights/danang_final.pth'))

# CHÈN THÊM ĐOẠN NÀY ĐỂ HIỂN THỊ LÊN GIAO DIỆN:
        st.sidebar.success("🧠 Bộ não: Đã nạp thành công trọng số STGCN Đà Nẵng")
        st.sidebar.caption("• Cấu trúc đồ thị: Đồ thị gốc (Original Graph)")
        st.sidebar.caption("• Không gian: 2,891 Ngã tư/Giao lộ (5 Quận)")
        st.sidebar.caption("• Trọng số khả huấn: Tích chập phổ Chebyshev bậc K=3")
        adj_data = np.load('dist_mat_danang_v2.npz')
        A_danang = torch.from_numpy(adj_data['adj_matrix']).float().to(device)
        
        mean_std = np.load('traffic_mean_std.npy')
        traffic_mean, traffic_std = mean_std[0], mean_std[1]
        
        return model, A_danang, traffic_mean, traffic_std, device
    except Exception as e:
        return None, None, None, None, None

model, A_danang, traffic_mean, traffic_std, device = load_ai_system()

# ==========================================
# CẤU HÌNH TRANG GIAO DIỆN
# ==========================================
st.set_page_config(page_title="KINDLOGISTICS Operations", layout="wide", page_icon="🚚")

if 'da_du_bao' not in st.session_state:
    st.session_state.da_du_bao = False

st.title("🚚 KINDLOGISTICS - Hệ Thống Điều Phối Nhu Cầu Giao Nhận")
st.markdown("Hệ thống AI dự báo Không gian - Thời gian thực tế dựa trên luồng dữ liệu lịch sử.")

if model is None:
    st.error("❌ LỖI HỆ THỐNG: Không tìm thấy file trọng số hoặc cấu hình mạng lưới (.pth, .npz, .npy).")
    st.stop()

# ==========================================
# 1. SIDEBAR: BẢNG ĐIỀU KHIỂN DỮ LIỆU & KỊCH BẢN
# ==========================================
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    
    # Gợi ý sửa lại text trên giao diện Streamlit cho chuẩn bài
st.markdown("### 📂 1. Dữ liệu Lịch sử Thực tế")
uploaded_file = st.file_uploader(
    "Tải file dữ liệu 4 bước thời gian - 1 tiếng quá khứ (CSV)", 
    type=["csv"]
)
st.caption("Khuyến nghị: File dung lượng < 5MB (Chứa thông tin 2.891 ngã tư)")
    
    if history_file:
        st.success(f"✅ Đã nhận luồng dữ liệu đầu vào.")
    else:
        # SỬA LỖI: Nếu gỡ file, reset toàn bộ trạng thái hệ thống ngay lập tức
        st.session_state.da_du_bao = False
        st.warning("⚠️ Cần nạp file dữ liệu lịch sử để kích hoạt mô hình AI.")

    st.subheader("⏳ 2. Tầm nhìn dự báo")
    time_ahead = st.slider("Dự báo trước (Phút)", min_value=5, max_value=60, value=15, step=5)

    st.subheader("🌦️ 3. Tương tác Kịch bản Ngoại cảnh")
    temp = st.slider("Nhiệt độ môi trường (°C)", 15, 45, 28) 
    rain = st.slider("Lượng mưa đo được (mm)", 0, 150, 0)
    
    st.markdown("**Yếu tố hiệu chỉnh nghiệp vụ:**")
    is_holiday = st.checkbox("Ngày Lễ/Tết (+30% nhu cầu)")
    is_sale = st.checkbox("Ngày Sale lớn (+50% nhu cầu)")
    
    st.divider()
    predict_btn = st.button("🚀 KÍCH HOẠT MẠNG GNN", use_container_width=True, disabled=(history_file is None))
    if predict_btn:
        st.session_state.da_du_bao = True

# ==========================================
# 2. XỬ LÝ AI LÀN TRUYỀN & HIỂN THỊ KẾT QUẢ
# ==========================================
col_map, col_info = st.columns([3, 1])

# Tạo mã danh tính (ID Key) động cho bản đồ dựa trên trạng thái của file đầu vào
# Kỹ thuật này ép Streamlit phải xóa sạch bản đồ cũ khi file bị gỡ hoặc thay đổi
map_key = "map_empty"
if history_file is not None and st.session_state.da_du_bao:
    map_key = f"map_populated_{history_file.name}_{rain}_{temp}_{time_ahead}_{is_sale}_{is_holiday}"

with col_map:
    m = folium.Map(location=[16.0544, 108.2022], zoom_start=13, tiles="CartoDB dark_matter")

    if st.session_state.da_du_bao and history_file is not None:
        try:
            # 🚀 KHỞI TẠO THANH THEO DÕI TIẾN TRÌNH DATA PIPELINE
            tien_trinh = st.progress(0, text="⏳ Khởi động ống dẫn dữ liệu (Data Pipeline)...")
            
            df_coords = pd.read_csv('danang_nodes_coords.csv')
            df_history = pd.read_csv(history_file)
            
            # Bước 1: Đếm dòng thực tế trong file
            so_dong_thuc_te = len(df_history)
            tien_trinh.progress(25, text=f"📂 Đã đọc xong file CSV. Phát hiện hệ thống có: **{so_dong_thuc_te:,} dòng**")
            
            feature_cols = ['traffic', 'feat2', 'feat3', 'temp', 'rain', 'feat6']
            expected_rows = 12 * 2891
            
            # Bước 2: Kiểm tra tính toán phân rã Không gian - Thời gian
            tien_trinh.progress(50, text=f"🔍 Đang phân rã dữ liệu: Yêu cầu {expected_rows} dòng (12 khung giờ × 2.891 ngã tư)...")
            
            raw_data = df_history[feature_cols].iloc[:expected_rows].values 
            reshaped_data = raw_data.reshape(12, 2891, 6)
            normalized_data = np.copy(reshaped_data)
            
            # Bước 3: Chuẩn hóa Z-Score và nạp ma trận kề
            tien_trinh.progress(75, text="🧠 Đang đồng bộ hóa ma trận kề Laplacian và chuẩn hóa Z-score các yếu tố ngoại cảnh...")
            
            normalized_data[:, :, 0] = (reshaped_data[:, :, 0] - traffic_mean) / (traffic_std + 1e-5)
            normalized_data[:, :, 3] = (temp - 28) / 5.0  
            normalized_data[:, :, 4] = rain / 20.0        
            
            x_input = torch.from_numpy(normalized_data).float()
            x_input = x_input.permute(2, 1, 0).unsqueeze(0).to(device)
            
            # Bước 4: Kích hoạt mô hình AI dự báo
            tien_trinh.progress(90, text="🚀 Đang nạp Tensor vào mạng STGCN (Tích chập đa thức Chebyshev)...")
            
            with torch.no_grad():
                out = model(x_input, A_danang) 
            
            pred_zscore = out.mean(dim=2).squeeze().cpu().numpy()
            pred_real = (pred_zscore * traffic_std) + traffic_mean
            
            # Hậu xử lý
            multiplier = 1.0
            if is_holiday: multiplier += 0.3
            if is_sale: multiplier += 0.5
            penalty = (rain * 0.1) + (time_ahead * 0.05) 
            
            final_predictions = (pred_real + penalty) * multiplier
            final_predictions = np.clip(final_predictions, 5, 450) 

            # Bước 5: Hoàn tất và dựng bản đồ
            tien_trinh.progress(100, text="🟢 Xử lý toán học thành công! Đang dựng bản đồ nhiệt Heatmap Đà Nẵng...")
            
            # Rải Heatmap lên bản đồ
            heat_data = [[df_coords.iloc[i]['lat'], df_coords.iloc[i]['lon'], float(final_predictions[i])] for i in range(len(df_coords))]
            HeatMap(heat_data, radius=14, blur=9, min_opacity=0.3,
                    gradient={0.4: 'blue', 0.6: 'cyan', 0.8: 'yellow', 1: 'red'}).add_to(m)
            
            if rain > 50:
                folium.Marker([16.061, 108.211], popup="Ngập cục bộ: Hàm Nghi", icon=folium.Icon(color='red', icon='cloud')).add_to(m)
                folium.Marker([16.065, 108.215], popup="Ngập cục bộ: Thạc Gián", icon=folium.Icon(color='red', icon='cloud')).add_to(m)

            # Xóa thanh trạng thái sau khi đã dựng bản đồ xong cho gọn giao diện
            tien_trinh.empty()

        except Exception as e:
            st.error(f"❌ Lỗi xử lý toán học: {e}")
            
            #if len(df_history) < expected_rows:
              #  st.error(f"❌ File yêu cầu chính xác {expected_rows} dòng.")
              #  st.stop()
                
            raw_data = df_history[feature_cols].iloc[:expected_rows].values 
            reshaped_data = raw_data.reshape(12, 2891, 6)
            normalized_data = np.copy(reshaped_data)
            
            # Tiến hành chuẩn hóa dữ liệu thật đưa vào mạng
            normalized_data[:, :, 0] = (reshaped_data[:, :, 0] - traffic_mean) / (traffic_std + 1e-5)
            normalized_data[:, :, 3] = (temp - 28) / 5.0  
            normalized_data[:, :, 4] = rain / 20.0        
            
            x_input = torch.from_numpy(normalized_data).float()
            x_input = x_input.permute(2, 1, 0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(x_input, A_danang) 
            
            pred_zscore = out.mean(dim=2).squeeze().cpu().numpy()
            pred_real = (pred_zscore * traffic_std) + traffic_mean
            
            # Hậu xử lý
            multiplier = 1.0
            if is_holiday: multiplier += 0.3
            if is_sale: multiplier += 0.5
            penalty = (rain * 0.1) + (time_ahead * 0.05) 
            
            final_predictions = (pred_real + penalty) * multiplier
            final_predictions = np.clip(final_predictions, 5, 450) 

            # Rải Heatmap lên bản đồ
            heat_data = [[df_coords.iloc[i]['lat'], df_coords.iloc[i]['lon'], float(final_predictions[i])] for i in range(len(df_coords))]
            HeatMap(heat_data, radius=14, blur=9, min_opacity=0.3,
                    gradient={0.4: 'blue', 0.6: 'cyan', 0.8: 'yellow', 1: 'red'}).add_to(m)
            
            if rain > 50:
                folium.Marker([16.061, 108.211], popup="Ngập cục bộ: Hàm Nghi", icon=folium.Icon(color='red', icon='cloud')).add_to(m)
                folium.Marker([16.065, 108.215], popup="Ngập cục bộ: Thạc Gián", icon=folium.Icon(color='red', icon='cloud')).add_to(m)

        except Exception as e:
            st.error(f"❌ Lỗi xử lý toán học: {e}")

    # Đưa key động vào đây để triệt tiêu bộ nhớ đệm Folium
    st_folium(m, width=950, height=650, returned_objects=[], key=map_key)

# ==========================================
# 3. HIỂN THỊ CHỈ SỐ VÀ KHUYẾN NGHỊ VẬN HÀNH
# ==========================================
with col_info:
    st.subheader("📝 Phân tích Vận hành")
    if st.session_state.da_du_bao and history_file is not None:
        
        # 🔍 ĐOẠN DEBUG THỰC TẾ: Hiện chỉ số thô của file nạp vào để kiểm chứng
        st.info(f"📊 Đang đọc file: **{history_file.name}**\n\nGiá trị gốc trung bình trong file CSV: `{raw_data[:, 0].mean():.2f}`")
        
        st.divider()
        st.write("**Trạng thái mạng lưới:**")
        if is_sale: st.error("🔥 CẢNH BÁO: QUÁ TẢI NGÀY SALE")
        elif is_holiday: st.warning("🎊 CẢNH BÁO: NHU CẦU NGÀY LỄ")
        else: st.success("🟢 Vận hành ổn định")

        st.divider()
        st.write(f"**Dự báo điểm nút ({time_ahead} phút tới):**")
        st.markdown(f"- **Khu vực Hàm Nghi:** ~{int(final_predictions[100] if len(final_predictions)>100 else 85)} đơn/h")
        st.markdown(f"- **Khu vực Thạc Gián:** ~{int(final_predictions[500] if len(final_predictions)>500 else 60)} đơn/h")
    else:
        st.info("Hệ thống đang ở trạng thái tĩnh. Hãy nạp luồng dữ liệu CSV lịch sử để kích hoạt phân tích.")
