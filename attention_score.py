import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

try:
    from models.TransMIL import TransMIL
except ImportError:
    print("Error: Could not import TransMIL. Please ensure this script is in the TransMIL project root or adjust the python path.")
    # Dummy class for syntax checking if import fails
    class TransMIL(nn.Module):
        def __init__(self, **kwargs): super().__init__()

class BagDataset(Dataset):

    def __init__(self, feature_dir, csv_path=None):
        self.feature_dir = feature_dir
        if csv_path:
            self.slide_ids = pd.read_csv(csv_path)['slide_id'].tolist()
        else:
            self.slide_ids = [f.replace('.pt', '') for f in os.listdir(feature_dir) if f.endswith('.pt')]
        
    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        feature_path = os.path.join(self.feature_dir, f"{slide_id}.pt")
        
        features = torch.load(feature_path)
        
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
            
        return features, slide_id

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', type=str, required=True, help='')
    parser.add_argument('--feature_dir', type=str, required=True, help='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--n_classes', type=int, default=2, help='')
    parser.add_argument('--device', type=str, required=True)
    return parser.parse_args()

attention_weights = []

def hook_fn(module, input, output):

    pass 

def generate_scores():
    args = get_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    model = TransMIL(n_classes=args.n_classes).to(args.device)

    checkpoint = torch.load(args.model_path, map_location=args.device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('model.', '')] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    dataset = BagDataset(feature_dir=args.feature_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) 
    
    print(f"Start processing {len(dataset)} slides...")
    
    with torch.no_grad():
        for features, slide_ids in tqdm(loader):
            features = features.to(args.device) # [1, N, 1024]
            slide_id = slide_ids[0]
            x = features
            
            #Projection & PGL
            x = model.pos_layer(x, features) # [1, N, 512]
            x = model.layer1(x) # [1, N, 512]
            
            output_layer2 = model.layer2(x) # [1, N+1, 512]

            output = model.norm(output_layer2)

            cls_token = output[:, 0, :] # [1, 512]
            patch_tokens = output[:, 1:, :] # [1, N, 512]
            
            # Normalize vectors
            cls_norm = torch.nn.functional.normalize(cls_token, p=2, dim=1)
            patch_norm = torch.nn.functional.normalize(patch_tokens, p=2, dim=2)

            attn_scores = torch.sum(cls_norm.unsqueeze(1) * patch_norm, dim=-1).squeeze(0) # [N]

            attn_scores_np = attn_scores.cpu().numpy()

            df = pd.DataFrame({
                'patch_id': range(len(attn_scores_np)),
                'attention_score': attn_scores_np
            })
            
            save_path = os.path.join(args.output_dir, f"{slide_id}.csv")
            df.to_csv(save_path, index=False)

    print("Done! Attention scores saved.")

if __name__ == "__main__":
    generate_scores()