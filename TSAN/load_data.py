import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd

meta_dir = '/home/canhdx/workspace/TSAN-brain-age-estimation/label.xlsx'
# data_dir = '/media/sslab/PACS/sslab/letuananh/fl-mri-age-prediction/server_data_fsl_TEST/sub-BrainAge000050.npy'

def nii_loader(path):
    # img = nib.load(str(path))
    # data = img.get_fdata()
    data = np.load(path)
    return data

def read_table(path):
    return(pd.read_excel(path).values) # default to first sheet

class Integer_Multiple_Batch_Size(torch.utils.data.Dataset):
    
    def __init__(self, folder_dataset, batch_size=8):
        self.folder_dataset = folder_dataset
        self.batch_size = batch_size

        source_dataset_len = len(self.folder_dataset)
        num_need_to_complement = self.batch_size - (source_dataset_len % self.batch_size)
        
        idx_list = np.arange(0, source_dataset_len)
        complement_idx = idx_list[-num_need_to_complement:]
        self.complemented_idx = np.concatenate([idx_list, complement_idx], axis=0)
        self.complemented_size = self.complemented_idx.shape[0]
        print(self.complemented_idx.shape, self.complemented_size)
        
    def __len__(self):
        return self.complemented_size

    def __getitem__(self, index):
        return self.folder_dataset[self.complemented_idx[index]]
    
class IMG_Folder(torch.utils.data.Dataset):
    def __init__(self,excel_path, data_path, loader=nii_loader,transforms=None):
        self.root = data_path
        self.sub_fns = sorted(os.listdir(self.root))
        self.table_refer = read_table(excel_path)
        self.loader = loader
        self.transform = transforms

    def __len__(self):
        return len(self.sub_fns)

    def __getitem__(self,index):
        sub_fn = self.sub_fns[index]
        for f in self.table_refer:
            
            sid = str(f[3])
            slabel = (int(f[0]))
            smale = f[2]
            if sid not in sub_fn:
                continue
            sub_path = os.path.join(self.root, sub_fn)
            img = self.loader(sub_path)
            if self.transform is not None:
                img = self.transform(img)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img, dtype= np.float32)
            img = torch.from_numpy(img).type(torch.FloatTensor)

            break
        return (img, sid, slabel, smale)
