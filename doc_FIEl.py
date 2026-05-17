import numpy as np
import pandas as pd

def verify_danang_data(file_path=r'D:\CDIOFN\danang_1year_6features.npy'):
    print(f"🔍 Đang kiểm tra file: {file_path}...")
    
    try:
        # Load ma trận dữ liệu
        data = np.load(file_path)
        
        # 1. Kiểm tra Shape (Kích thước)
        # Kỳ vọng: (Thời gian, 2891 nút, 6 đặc trưng)
        shape = data.shape
        print(f"📏 Kích thước Tensor: {shape}")
        
        if len(shape) == 3 and shape[2] == 6:
            print("✅ XÁC NHẬN: File chứa đúng 6 đặc trưng (Channels).")
        else:
            print(f"❌ CẢNH BÁO: Số lượng đặc trưng hiện tại là {shape[2]}, không phải 6!")

        # 2. Giải mã chi tiết từng đặc trưng
        feature_names = [
            "Lưu lượng (Traffic)", 
            "Lượng mưa (Rainfall)", 
            "Mức độ ngập (Flood)", 
            "Nhiệt độ (Temp)", 
            "Độ ẩm (Humidity)", 
            "Tốc độ (Speed)"
        ]
        
        print("\n=== PHÂN TÍCH GIÁ TRỊ TỪNG ĐẶC TRƯNG ===")
        for i in range(shape[2]):
            # Lấy toàn bộ dữ liệu của đặc trưng thứ i
            feature_slice = data[:, :, i]
            
            mean_val = np.mean(feature_slice)
            max_val = np.max(feature_slice)
            min_val = np.min(feature_slice)
            
            print(f"Feature {i} [{feature_names[i]}]:")
            print(f"   - Trung bình: {mean_val:.2f}")
            print(f"   - Cao nhất:  {max_val:.2f}")
            print(f"   - Thấp nhất: {min_val:.2f}")
            print("-" * 30)

    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file. Ông kiểm tra lại đường dẫn nhé!")
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")

if __name__ == "__main__":
    verify_danang_data()

# Đọc file CSV chứa tọa độ nodes Đà Nẵng
df_coords = pd.read_csv('danang_nodes_coords.csv')

# Hiển thị 5 dòng đầu để kiểm tra
print(df_coords.head())

# Nếu cần truy cập cột cụ thể (giả sử có cột 'node_id', 'lat', 'lon')
# print(df_coords[['node_id', 'lat', 'lon']])