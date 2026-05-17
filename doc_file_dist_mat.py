import numpy as np

# Đường dẫn file
npz_path = 'data/cd/dist_mat.npz'

try:
    data = np.load(npz_path)
    print("--- THÔNG TIN FILE dist_mat.npz ---")
    print(f"Các khóa (Keys) bên trong: {data.files}")
    
    # Giả sử khóa đầu tiên chứa ma trận
    matrix_key = data.files[0]
    matrix = data[matrix_key]
    
    print(f"Kích thước ma trận (Shape): {matrix.shape}")
    print(f"Kiểu dữ liệu: {matrix.dtype}")
    print(f"5 giá trị đầu tiên: {matrix[:5]}")
    
    # Gợi ý viết báo cáo: 
    # Nếu shape là (43105780,), hãy giải thích đây là ma trận phẳng 
    # của 6566 x 6566 đoạn đường.
except FileNotFoundError:
    print("Không tìm thấy file dist_mat.npz")