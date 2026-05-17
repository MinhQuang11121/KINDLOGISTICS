import torch
import torch.nn as nn
import torch.nn.functional as F

class STConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.temp_conv1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size=(3, 1), padding=(1, 0))
        self.spatial_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        self.temp_conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=(3, 1), padding=(1, 0))
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, A):
        x1 = self.temp_conv1(x)
        x1_1, x1_2 = torch.split(x1, x1.size(1) // 2, dim=1)
        x1 = x1_1 * torch.sigmoid(x1_2)
        
        x2 = self.spatial_conv(x1)
        x2 = torch.einsum('bctn, nm -> bctm', x2, A)
        x2 = F.relu(x2)
        
        x3 = self.temp_conv2(x2)
        x3_1, x3_2 = torch.split(x3, x3.size(1) // 2, dim=1)
        x3 = x3_1 * torch.sigmoid(x3_2)
        return self.batch_norm(x3)

class STGCN(nn.Module):
    # ĐÃ THÊM num_layers VÀO ĐÂY ĐỂ TRÁNH LỖI
    def __init__(self, num_nodes, in_features, hidden_features, seq_len, pre_len, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.layers = nn.ModuleList()
        
        # Tầng 1
        self.layers.append(STConvBlock(in_features, hidden_features))
        # Các tầng tiếp theo
        for _ in range(num_layers - 1):
            self.layers.append(STConvBlock(hidden_features, hidden_features))
            
        self.final_conv = nn.Conv2d(hidden_features, pre_len, kernel_size=(seq_len, 1))

    def forward(self, x, A):
        x = x.permute(0, 1, 3, 2) 
        x = x[:, :, :, :self.num_nodes]
        
        # Vòng lặp này giúp AI học qua nhiều tầng tư duy (3 block)
        for layer in self.layers:
            x = layer(x, A)
            
        out = self.final_conv(x)
        out = out.squeeze(2)
        return out.permute(0, 2, 1)