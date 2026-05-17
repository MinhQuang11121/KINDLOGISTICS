import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from model import STGCN

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="AI Logistics Da Nang", layout="wide")
st.title("📊 Hệ Thống Dự Báo Nhu Cầu Logistics Đà Nẵng (STGCN)")
st.markdown("---")

# --- 1. NẠP DỮ LIỆU & MODEL ---
@st.cache_resource
def load_model_and_data():
    num_nodes = 2891
    in_features = 6
    device = torch.device("cpu")
    
    # Nạp Model
    model = STGCN(num_nodes, in_features, 128, 12, 3, 3)
    model.load_state_dict(torch.load('models_weights/danang_ultra_v6.pth', map_location=device))
    model.eval()
    
    # Nạp Data Test (để demo)
    data = np.load('danang_1year_6features.npy')
    # Nạp tọa độ ngã tư (Ông cần file csv chứa lat, lon của 2891 node này nhé)
    # df_coords = pd.read_csv('danang_nodes_coords.csv') 
    
    return model, data

model, data = load_model_and_data()

# --- 2. THANH SIDEBAR ĐIỀU KHIỂN ---
st.sidebar.header("🕹️ Bảng Điều Khiển")
node_id = st.sidebar.number_input("Chọn mã Ngã tư (Node ID):", min_value=0, max_value=2890, value=100)
time_slot = st.sidebar.slider("Chọn mốc thời gian demo:", 0, 1000, 500)

# --- 3. HIỂN THỊ CHỈ SỐ NHANH ---
col1, col2, col3, col4 = st.columns(4)
current_info = data[time_slot, node_id, :]

with col1:
    st.metric("Nhu cầu hiện tại", f"{int(current_info[0])} đơn")
with col2:
    st.metric("Nhiệt độ", f"{current_info[1]:.1f} °C")
with col3:
    st.metric("Tình trạng", "Mưa" if current_info[2] > 0 else "Nắng")
with col4:
    status = "Bình thường"
    if current_info[5] == 1: status = "SỰ KIỆN SALE"
    if current_info[5] == 2: status = "NGÀY LỄ"
    st.metric("Dịp đặc biệt", status)

# --- 4. DỰ BÁO VỚI AI ---
st.subheader(f"📈 Dự báo nhu cầu tại Ngã tư #{node_id}")

# Lấy 12 khung quá khứ
input_seq = data[time_slot-12 : time_slot, :, :]
input_tensor = torch.from_numpy(input_seq).unsqueeze(0).permute(0, 3, 2, 1).float()

# Nạp bản đồ A (cần nạp lại file npz)
adj_data = np.load('dist_mat_danang_v2.npz')
A = torch.from_numpy(adj_data['adj_matrix']).float()

with torch.no_grad():
    prediction = model(input_tensor, A).squeeze().numpy() # [3, 2891]
    node_pred = prediction[:, node_id]

# Vẽ biểu đồ So sánh
fig = go.Figure()
# Thực tế (Quá khứ)
fig.add_trace(go.Scatter(x=list(range(-11, 1)), y=input_seq[:, node_id, 0], name="Quá khứ (Thật)", line=dict(color='blue')))
# Dự báo (Tương lai)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=node_pred, name="AI Dự báo", line=dict(color='red', dash='dash')))

fig.update_layout(xaxis_title="Thời gian (mỗi 15 phút)", yaxis_title="Số lượng đơn hàng")
st.plotly_chart(fig, use_container_width=True)

# --- 5. BẢN ĐỒ NHIỆT (HEATMAP) ---
st.subheader("📍 Bản đồ nhiệt Logistics toàn thành phố")
st.info("Phần này cần file tọa độ CSV để hiển thị chính xác vị trí 2891 ngã tư.")
# df_map = df_coords.copy()
# df_map['demand'] = data[time_slot, :, 0]
# st.map(df_map) # Streamlit tự vẽ bản đồ nếu có cột lat, lon