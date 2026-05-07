import os
import glob
import torch
from torch.utils.data import Dataset
# [新增] 必须导入这个 Data 类
from torch_geometric.data import Data 

class MoleculeDataset(Dataset):
    def __init__(self, root):
        self.files = glob.glob(os.path.join(root, "*.pt"))
        if len(self.files) == 0:
            print(f"Warning: No .pt files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # 1. 加载字典
        data_dict = torch.load(self.files[i])
        
        # 2. [关键修改] 将字典转换为 PyG 的 Data 对象
        # 只有变成了 Data 对象，DataLoader 才知道要把它们拼成一个大图，而不是堆叠
        data = Data(
            x = data_dict['x'],
            pos = data_dict['pos'],
            mask_type = data_dict['mask_type']
        )
        
        # 3. 把标签也挂载上去 (根据是否存在来挂载)
        if 'y_fg' in data_dict:
            data.y_fg = data_dict['y_fg']
        if 'y_angle' in data_dict:
            data.y_angle = data_dict['y_angle']
        if 'y_fp' in data_dict:
            # 注意：指纹通常是 [1, 2048] 或 [2048]，确保维度正确
            data.y_fp = data_dict['y_fp'].view(1, -1) 
        if 'y_affinity' in data_dict:
            data.y_affinity = data_dict['y_affinity']
            
        return data