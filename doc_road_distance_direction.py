import pandas as pd
import numpy as np

def analyze_road_data(file_path=r'D:\CDIOFN\data\cd\road_distance_direction.txt'):
    print(f"🚀 Đang phân tích dữ liệu không gian: {file_path}...")
    
    try:
        # Vì file 1.1GB, ta chỉ đọc 1000 dòng đầu để lấy mẫu mô tả
        # File không có tiêu đề nên ta tự gán tên cột
        column_names = ['source_node', 'target_node', 'distance_m']
        df = pd.read_csv(file_path, header=None, names=column_names, nrows=1000)
        
        print("\n=== CẤU TRÚC 5 DÒNG DỮ LIỆU ĐẦU TIÊN ===")
        print(df.head())
        
        # Thống kê nhanh
        inf_count = (df['distance_m'] == 'inf').sum() + df['distance_m'].isna().sum()
        print(f"\n📊 Thống kê mẫu: {len(df)} cặp nút")
        print(f"🔗 Số lượng cặp nút không có kết nối (inf): {inf_count}")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    analyze_road_data()