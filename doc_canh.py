import numpy as np

adj_data = np.load('dist_mat_danang_v2.npz')
matrix = adj_data['adj_matrix']

# Kiểm tra ma trận có đối xứng không
is_symmetric = np.allclose(matrix, matrix.T)
print("Ma trận kề có đối xứng không:", is_symmetric)

# Tính bậc trung bình của các nút
degrees = np.sum(matrix > 0, axis=1)
print("Bậc lớn nhất của một nút:", np.max(degrees))
print("Bậc trung bình của các nút:", np.mean(degrees))