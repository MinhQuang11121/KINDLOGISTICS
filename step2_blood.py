import os
import pandas as pd
import numpy as np
import ast

# 1. Cấu hình
base_path = r'D:\CDIOFN'
path_csv = os.path.join(base_path, 'data', 'cd', 'filtered_lines.csv') 
path_out = os.path.join(base_path, 'v_data.npy')
num_nodes = 6566
time_window = 300 # 5 phút = 300 giây

print("⏳ Bước 2: Đang trích xuất lưu lượng chuẩn từ dấu chấm phẩy (;)...")

# 2. Đọc file với dấu ngăn cách chuẩn
# Dùng usecols để chỉ lấy những gì cần thiết, đỡ nặng RAM
df = pd.read_csv(path_csv, sep=';', usecols=['path', 'timestamp'])

# 3. Lấy mốc thời gian bắt đầu của toàn bộ dataset để làm mốc 0
# Vì timestamp là chuỗi "[...]", ta cần lấy số đầu tiên trong chuỗi đó
def get_first_ts(ts_str):
    try:
        return ast.literal_eval(ts_str)[0]
    except:
        return None

print("📊 Đang tính toán mốc thời gian...")
first_timestamps = df['timestamp'].apply(get_first_ts).dropna()
global_start_ts = first_timestamps.min()

# 4. Tạo ma trận rỗng
max_ts = first_timestamps.max()
max_time_idx = int((max_ts - global_start_ts) // time_window) + 1
v_matrix = np.zeros((max_time_idx + 1, num_nodes))

print(f"🚀 Đang xử lý {len(df)} hành trình. Đợi card RTX A4000 tí nhé...")

# 5. Xay dữ liệu: Mỗi đoạn đường trong path ứng với 1 timestamp tương ứng
for _, row in df.iterrows():
    try:
        path_list = ast.literal_eval(row['path'])
        ts_list = ast.literal_eval(row['timestamp'])
        
        for i in range(len(path_list)):
            edge_id = path_list[i]
            ts = ts_list[i]
            
            if edge_id < num_nodes:
                t_idx = int((ts - global_start_ts) // time_window)
                if 0 <= t_idx < v_matrix.shape[0]:
                    v_matrix[t_idx, edge_id] += 1
    except:
        continue

# 6. Lưu thành phẩm
np.save(path_out, v_matrix)
print(f"✅ Xong! Đã tạo ra 'máu' xịn tại: {path_out}")
print(f"📈 Kích thước ma trận: {v_matrix.shape}")