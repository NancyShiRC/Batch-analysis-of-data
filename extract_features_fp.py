import logging
from datetime import datetime
import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

# 初始化日志系统
def setup_logging(feat_dir):
    """配置日志系统，同时输出到文件和控制台"""
    os.makedirs(feat_dir, exist_ok=True)
    log_dir = os.path.join(feat_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'feature_extraction_{timestamp}.log')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 文件handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return log_file

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(output_path, loader, model, verbose=0):
    """[原有代码保持不变，添加日志]"""
    if verbose > 0:
        logging.info(f'Processing a total of {len(loader)} batches')
    
    mode = 'w'
    for count, data in enumerate(tqdm(loader, desc="Processing batches")):
        with torch.inference_mode():    
            batch = data['img']
            coords = data['coord'].numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True)
            
            features = model(batch)
            features = features.cpu().numpy().astype(np.float32)

            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
    
    logging.info(f'Completed processing for {output_path}')
    return output_path

def is_processing_complete(feat_dir, slide_id):
    """检查该slide是否已经完整处理"""
    pt_file = os.path.join(feat_dir, 'pt_files', f'{slide_id}.pt')
    h5_file = os.path.join(feat_dir, 'h5_files', f'{slide_id}.h5')
    
    # 两个文件都存在且非空
    complete = os.path.exists(pt_file) and os.path.getsize(pt_file) > 0 and \
               os.path.exists(h5_file) and os.path.getsize(h5_file) > 0
    if complete:
        logging.debug(f'Slide {slide_id} already processed')
    return complete

def validate_h5_file(h5_path):
    """验证h5文件完整性"""
    try:
        with h5py.File(h5_path, 'r') as f:
            valid = 'features' in f and 'coords' in f
        if not valid:
            logging.warning(f'Invalid HDF5 file structure in {h5_path}')
        return valid
    except Exception as e:
        logging.error(f'Failed to validate HDF5 file {h5_path}: {str(e)}')
        return False

def main():
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--data_h5_dir', type=str, required=True)
    parser.add_argument('--data_slide_dir', type=str, required=True)
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--feat_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='resnet50_trunc', 
                       choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--log_level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    # 设置日志
    log_file = setup_logging(args.feat_dir)
    logging.info(f'Starting feature extraction, logging to {log_file}')
    logging.info(f'Command line arguments: {vars(args)}')

    try:
        logging.info('Initializing dataset')
        bags_dataset = Dataset_All_Bags(args.csv_path)
        
        os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
        os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)

        logging.info(f'Loading model {args.model_name}')
        model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
        model.eval()
        model = model.to(device)
        total = len(bags_dataset)
        logging.info(f'Total slides to process: {total}')

        loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

        for bag_candidate_idx in range(total):
            slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
            bag_name = f'{slide_id}.h5'
            
            logging.info(f'\nProcessing slide {bag_candidate_idx+1}/{total}: {slide_id}')

            # 检查是否应该跳过
            if not args.no_auto_skip and is_processing_complete(args.feat_dir, slide_id):
                logging.info(f'Skipping {slide_id} (already processed)')
                continue

            h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
            slide_file_path = os.path.join(args.data_slide_dir, f'{slide_id}{args.slide_ext}')
            output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)

            # 检查输入文件
            if not os.path.exists(slide_file_path):
                logging.error(f'Slide file {slide_file_path} not found, skipping')
                continue

            start_time = time.time()
            try:
                logging.info(f'Opening slide {slide_file_path}')
                wsi = openslide.open_slide(slide_file_path)
                
                logging.info(f'Creating dataset for {slide_id}')
                dataset = Whole_Slide_Bag_FP(
                    file_path=h5_file_path, 
                    wsi=wsi, 
                    img_transforms=img_transforms
                )

                loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
                logging.info(f'Starting feature computation for {slide_id}')
                
                output_file_path = compute_w_loader(output_path, loader=loader, model=model, verbose=1)

                elapsed_time = time.time() - start_time
                logging.info(f'Feature computation for {slide_id} completed in {elapsed_time:.2f} seconds')

                # 验证输出文件
                if validate_h5_file(output_file_path):
                    with h5py.File(output_file_path, "r") as file:
                        features = file['features'][:]
                        logging.info(f'Features size: {features.shape}')
                        logging.info(f'Coordinates size: {file["coords"].shape}')

                    features = torch.from_numpy(features)
                    torch.save(features, os.path.join(args.feat_dir, 'pt_files', f'{slide_id}.pt'))
                    logging.info(f'Saved features for {slide_id}')
                else:
                    logging.error(f'Invalid h5 file generated for {slide_id}, removing...')
                    if os.path.exists(output_file_path):
                        os.remove(output_file_path)

            except Exception as e:
                logging.error(f'Error processing {slide_id}: {str(e)}', exc_info=True)
                # 清理可能生成的不完整文件
                if os.path.exists(output_path):
                    os.remove(output_path)
                pt_path = os.path.join(args.feat_dir, 'pt_files', f'{slide_id}.pt')
                if os.path.exists(pt_path):
                    os.remove(pt_path)

        logging.info('Feature extraction completed successfully')

    except Exception as e:
        logging.critical(f'Feature extraction failed: {str(e)}', exc_info=True)
        raise

if __name__ == '__main__':
    main()