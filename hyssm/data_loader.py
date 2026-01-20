# import logging
# import pickle
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset
# __all__ = ['MMDataLoader']
# logger = logging.getLogger('MMSA')

# class MMDataset(Dataset):
#     def __init__(self, args, mode='train'):
#         self.mode = mode
#         self.args = args
#         DATASET_MAP = {
#             'mosi': self.__init_mosi,
#             'mosei': self.__init_mosei,
#         }
#         DATASET_MAP[args['dataset_name']]()

#     def __init_mosi(self):
#         with open(self.args['featurePath'], 'rb') as f:
#             data = pickle.load(f)
#         if 'use_bert' in self.args and self.args['use_bert']:
#             self.text = data[self.mode]['text_bert'].astype(np.float32)
#         else:
#             self.text = data[self.mode]['text'].astype(np.float32)
#         self.vision = data[self.mode]['vision'].astype(np.float32)
#         self.audio = data[self.mode]['audio'].astype(np.float32)
#         self.raw_text = data[self.mode]['raw_text']
#         self.ids = data[self.mode]['id']

#         if self.args['feature_T'] != "":
#             with open(self.args['feature_T'], 'rb') as f:
#                 data_T = pickle.load(f)
#             if 'use_bert' in self.args and self.args['use_bert']:
#                 self.text = data_T[self.mode]['text_bert'].astype(np.float32)
#                 self.args['feature_dims'][0] = 768
#             else:
#                 self.text = data_T[self.mode]['text'].astype(np.float32)
#                 self.args['feature_dims'][0] = self.text.shape[2]
#         if self.args['feature_A'] != "":
#             with open(self.args['feature_A'], 'rb') as f:
#                 data_A = pickle.load(f)
#             self.audio = data_A[self.mode]['audio'].astype(np.float32)
#             self.args['feature_dims'][1] = self.audio.shape[2]
#         if self.args['feature_V'] != "":
#             with open(self.args['feature_V'], 'rb') as f:
#                 data_V = pickle.load(f)
#             self.vision = data_V[self.mode]['vision'].astype(np.float32)
#             self.args['feature_dims'][2] = self.vision.shape[2]

#         self.labels = {
#             'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
#         }

#         logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

#         if not self.args['need_data_aligned']:
#             if self.args['feature_A'] != "":
#                 self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
#             else:
#                 self.audio_lengths = data[self.mode]['audio_lengths']
#             if self.args['feature_V'] != "":
#                 self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
#             else:
#                 self.vision_lengths = data[self.mode]['vision_lengths']
#         self.audio[self.audio == -np.inf] = 0

#         if 'need_normalized' in self.args and self.args['need_normalized']:
#             self.__normalize()
    
#     def __init_mosei(self):
#         return self.__init_mosi()

#     def __init_sims(self):
#         return self.__init_mosi()

#     def __truncate(self):
#         def do_truncate(modal_features, length):
#             if length == modal_features.shape[1]:
#                 return modal_features
#             truncated_feature = []
#             padding = np.array([0 for i in range(modal_features.shape[2])])
#             for instance in modal_features:
#                 for index in range(modal_features.shape[1]):
#                     if((instance[index] == padding).all()):
#                         if(index + length >= modal_features.shape[1]):
#                             truncated_feature.append(instance[index:index+20])
#                             break
#                     else:                        
#                         truncated_feature.append(instance[index:index+20])
#                         break
#             truncated_feature = np.array(truncated_feature)
#             return truncated_feature
        
#         text_length, audio_length, video_length = self.args['seq_lens']
#         self.vision = do_truncate(self.vision, video_length)
#         self.text = do_truncate(self.text, text_length)
#         self.audio = do_truncate(self.audio, audio_length)

#     def __normalize(self):
#         self.vision = np.transpose(self.vision, (1, 0, 2))
#         self.audio = np.transpose(self.audio, (1, 0, 2))
#         self.vision = np.mean(self.vision, axis=0, keepdims=True)
#         self.audio = np.mean(self.audio, axis=0, keepdims=True)

#         self.vision[self.vision != self.vision] = 0
#         self.audio[self.audio != self.audio] = 0

#         self.vision = np.transpose(self.vision, (1, 0, 2))
#         self.audio = np.transpose(self.audio, (1, 0, 2))

#     def __len__(self):
#         return len(self.labels['M'])

#     def get_seq_len(self):
#         if 'use_bert' in self.args and self.args['use_bert']:
#             return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
#         else:
#             return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

#     def get_feature_dim(self):
#         return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

#     def __getitem__(self, index):
#         sample = {
#             'raw_text': self.raw_text[index],
#             'text': torch.Tensor(self.text[index]), 
#             'audio': torch.Tensor(self.audio[index]),
#             'vision': torch.Tensor(self.vision[index]),
#             'index': index,
#             'id': self.ids[index],
#             'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
#         } 
#         if not self.args['need_data_aligned']:
#             sample['audio_lengths'] = self.audio_lengths[index]
#             sample['vision_lengths'] = self.vision_lengths[index]
#         return sample

# def MMDataLoader(args, num_workers):

#     datasets = {
#         'train': MMDataset(args, mode='train'),
#         'valid': MMDataset(args, mode='valid'),
#         'test': MMDataset(args, mode='test')
#     }

#     if 'seq_lens' in args:
#         args['seq_lens'] = datasets['train'].get_seq_len() 

#     dataLoader = {
#         ds: DataLoader(datasets[ds],
#                        batch_size=args['batch_size'],
#                        num_workers=num_workers,
#                        shuffle=True)
#         for ds in datasets.keys()
#     }
    
#     return dataLoader

import logging
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

__all__ = ['MMDataLoader']
logger = logging.getLogger('MMSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        pt_path = "/root/FoldFish/dataset/CoraDiff-Dataset/MOSI/mosi_unified.pt"
        logger.info(f"Loading data from {pt_path} for mode {self.mode}...")
        
        data_dict = torch.load(pt_path, map_location='cpu')
        
        # 1. 建立全局查找表 (Global Lookup Tables)
        # 用于在 __getitem__ 中通过 Key 快速找到对应的特征
        self.global_zs_lookup = {}   # 存 LLM 语义特征 (Zs)
        self.global_phys_lookup = {} # 存 物理特征 (Zp source)
        
        for k, v in data_dict.items():
            # Zs: 直接取 LLM 提取的 hidden states [Dim=3584]
            self.global_zs_lookup[k] = {
                'text': v['text']['zs'],
                'audio': v['audio']['zs'],
                'visual': v['visual']['zs']
            }
            
            # Zp: 取物理特征，并做 Global Mean Pooling 变成向量 [SeqLen, Dim] -> [Dim]
            self.global_phys_lookup[k] = {
                'text': v['text']['feat'].float().mean(dim=0),   
                'audio': v['audio']['feat'].float().mean(dim=0), 
                'visual': v['visual']['feat'].float().mean(dim=0)
            }

        # 2. 筛选当前 Split 的数据
        self.data_list = []
        target_split = self.mode 
        for video_id, item in data_dict.items():
            if item['split'] == target_split:
                self.data_list.append(video_id)
        
        # 3. 预加载基础数据 (避免重复 IO)
        self.samples = {}
        for video_id in self.data_list:
            item = data_dict[video_id]
            self.samples[video_id] = {
                'text_feat': item['text']['feat'].numpy(),
                'audio_feat': item['audio']['feat'].numpy(),
                'vision_feat': item['visual']['feat'].numpy(),
                'label_reg': item['label_reg'].numpy(),
                'raw_text': item['annotation'],
                'retrieval': item['retrieval'] # 包含检索到的邻居 Key 和 Score
            }
            
        # 4. 更新 Config 中的维度 (防止写死导致报错)
        sample_id = self.data_list[0]
        self.args['feature_dims'][0] = self.samples[sample_id]['text_feat'].shape[1]
        self.args['feature_dims'][1] = self.samples[sample_id]['audio_feat'].shape[1]
        self.args['feature_dims'][2] = self.samples[sample_id]['vision_feat'].shape[1]
        
        logger.info(f"Loaded {len(self.data_list)} samples for {self.mode}. Dims: {self.args['feature_dims']}")

    def __init_mosei(self):
        pt_path = "/root/FoldFish/dataset/CoraDiff-Dataset/MOSEI/mosei_unified.pt"
        logger.info(f"Loading data from {pt_path} for mode {self.mode}...")
        
        data_dict = torch.load(pt_path, map_location='cpu')
        
        # 1. 建立全局查找表 (Global Lookup Tables)
        # 用于在 __getitem__ 中通过 Key 快速找到对应的特征
        self.global_zs_lookup = {}   # 存 LLM 语义特征 (Zs)
        self.global_phys_lookup = {} # 存 物理特征 (Zp source)
        
        for k, v in data_dict.items():
            # Zs: 直接取 LLM 提取的 hidden states [Dim=3584]
            self.global_zs_lookup[k] = {
                'text': v['text']['zs'],
                'audio': v['audio']['zs'],
                'visual': v['visual']['zs']
            }
            
            # Zp: 取物理特征，并做 Global Mean Pooling 变成向量 [SeqLen, Dim] -> [Dim]
            self.global_phys_lookup[k] = {
                'text': v['text']['feat'].float().mean(dim=0),   
                'audio': v['audio']['feat'].float().mean(dim=0), 
                'visual': v['visual']['feat'].float().mean(dim=0)
            }

        # 2. 筛选当前 Split 的数据
        self.data_list = []
        target_split = self.mode 
        for video_id, item in data_dict.items():
            if item['split'] == target_split:
                self.data_list.append(video_id)
        
        # 3. 预加载基础数据 (避免重复 IO)
        self.samples = {}
        for video_id in self.data_list:
            item = data_dict[video_id]
            self.samples[video_id] = {
                'text_feat': item['text']['feat'].numpy(),
                'audio_feat': item['audio']['feat'].numpy(),
                'vision_feat': item['visual']['feat'].numpy(),
                'label_reg': item['label_reg'].numpy(),
                'raw_text': item['annotation'],
                'retrieval': item['retrieval'] # 包含检索到的邻居 Key 和 Score
            }
            
        # 4. 更新 Config 中的维度 (防止写死导致报错)
        sample_id = self.data_list[0]
        self.args['feature_dims'][0] = self.samples[sample_id]['text_feat'].shape[1]
        self.args['feature_dims'][1] = self.samples[sample_id]['audio_feat'].shape[1]
        self.args['feature_dims'][2] = self.samples[sample_id]['vision_feat'].shape[1]
        
        logger.info(f"Loaded {len(self.data_list)} samples for {self.mode}. Dims: {self.args['feature_dims']}")

    def __init_chsims(self):
        pt_path = "/root/FoldFish/dataset/CoraDiff-Dataset/CH-SIMS/ch_sims_unified.pt"
        logger.info(f"Loading data from {pt_path} for mode {self.mode}...")
        
        data_dict = torch.load(pt_path, map_location='cpu')
        
        # 1. 建立全局查找表 (Global Lookup Tables)
        # 用于在 __getitem__ 中通过 Key 快速找到对应的特征
        self.global_zs_lookup = {}   # 存 LLM 语义特征 (Zs)
        self.global_phys_lookup = {} # 存 物理特征 (Zp source)
        
        for k, v in data_dict.items():
            # Zs: 直接取 LLM 提取的 hidden states [Dim=3584]
            self.global_zs_lookup[k] = {
                'text': v['text']['zs'],
                'audio': v['audio']['zs'],
                'visual': v['visual']['zs']
            }
            
            # Zp: 取物理特征，并做 Global Mean Pooling 变成向量 [SeqLen, Dim] -> [Dim]
            self.global_phys_lookup[k] = {
                'text': v['text']['feat'].float().mean(dim=0),   
                'audio': v['audio']['feat'].float().mean(dim=0), 
                'visual': v['visual']['feat'].float().mean(dim=0)
            }

        # 2. 筛选当前 Split 的数据
        self.data_list = []
        target_split = self.mode 
        for video_id, item in data_dict.items():
            if item['split'] == target_split:
                self.data_list.append(video_id)
        
        # 3. 预加载基础数据 (避免重复 IO)
        self.samples = {}
        for video_id in self.data_list:
            item = data_dict[video_id]
            self.samples[video_id] = {
                'text_feat': item['text']['feat'].numpy(),
                'audio_feat': item['audio']['feat'].numpy(),
                'vision_feat': item['visual']['feat'].numpy(),
                'label_reg': item['label_reg'].numpy(),
                'raw_text': item['annotation'],
                'retrieval': item['retrieval'] # 包含检索到的邻居 Key 和 Score
            }
            
        # 4. 更新 Config 中的维度 (防止写死导致报错)
        sample_id = self.data_list[0]
        self.args['feature_dims'][0] = self.samples[sample_id]['text_feat'].shape[1]
        self.args['feature_dims'][1] = self.samples[sample_id]['audio_feat'].shape[1]
        self.args['feature_dims'][2] = self.samples[sample_id]['vision_feat'].shape[1]
        
        logger.info(f"Loaded {len(self.data_list)} samples for {self.mode}. Dims: {self.args['feature_dims']}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        video_id = self.data_list[index]
        item = self.samples[video_id]
        
        # --- A. 基础数据 ---
        sample = {
            'index': index,
            'id': video_id,
            'text': torch.tensor(item['text_feat']), 
            'audio': torch.tensor(item['audio_feat']),
            'vision': torch.tensor(item['vision_feat']),
            'labels': {'M': torch.tensor(item['label_reg']).reshape(-1)},
            'raw_text': item['raw_text']
        }
        
        # --- B. 构建 Zs (Semantic Anchor) ---
        my_zs = self.global_zs_lookup[video_id]
        sample['zs'] = {
            'text': my_zs['text'],
            'audio': my_zs['audio'],
            'visual': my_zs['visual']
        }
        
        # --- C. 构建 Zp (Structural Prior via Retrieval) ---
        my_phys = self.global_phys_lookup[video_id] # 自己的物理特征作为兜底
        sample['zp'] = {}
        
        for m in ['text', 'audio', 'visual']:
            ret_info = item['retrieval'].get(m, None)
            
            # 如果没有检索结果，用自己兜底 (Self-Prior)
            if ret_info is None or len(ret_info['keys']) == 0:
                sample['zp'][m] = my_phys[m]
            else:
                neighbor_keys = ret_info['keys']
                scores = torch.tensor(ret_info['scores'])
                weights = F.softmax(scores, dim=0) # 归一化分数
                
                neighbor_feats = []
                for k in neighbor_keys:
                    # 查找邻居的物理特征
                    if k in self.global_phys_lookup:
                        neighbor_feats.append(self.global_phys_lookup[k][m])
                    else:
                        neighbor_feats.append(my_phys[m])
                
                neighbor_feats = torch.stack(neighbor_feats)
                # 加权平均: [K] @ [K, Dim] -> [Dim]
                zp_vec = torch.matmul(weights, neighbor_feats)
                sample['zp'][m] = zp_vec

        return sample

def MMDataLoader(args, num_workers):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }
    dataLoader = {
        ds: DataLoader(datasets[ds], batch_size=args['batch_size'], num_workers=num_workers, shuffle=True)
        for ds in datasets.keys()
    }
    return dataLoader