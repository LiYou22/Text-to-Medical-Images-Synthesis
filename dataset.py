import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import random
import xml.etree.ElementTree as ET
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IUXrayDataset(Dataset):
    def __init__(self, data_dir, image_size=256, split_ratio=0.9, is_train=True, max_samples=None):
        """
        Args:
            data_dir: Root directory containing the dataset
            image_size: Size to resize images to
            split_ratio: Ratio for train/test split
            is_train: Whether to use train or test split
            max_samples: Maximum number of samples to use (for debugging)
        """
        self.data_dir = data_dir
        self.is_train = is_train
        
        self.image_dir = os.path.join(data_dir, 'NLMCXR_png')
        self.report_dir = os.path.join(data_dir, 'ecgen-radiology')
        
        xml_files = glob.glob(os.path.join(self.report_dir, '*.xml'))
        
        self.samples = []
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                report_text = ""
                for text in root.findall('.//AbstractText'):
                    if text.text:
                        report_text += text.text + " "
                
                mesh_terms = []
                for mesh in root.findall('.//MeSH/*'): 
                    if mesh.text:
                        mesh_terms.append(mesh.text)
                
                findings = root.find('.//AbstractText[@Label="FINDINGS"]')
                impression = root.find('.//AbstractText[@Label="IMPRESSION"]')
                
                if findings is not None and findings.text and len(findings.text.strip()) > 20:
                    caption = findings.text.strip()
                elif impression is not None and impression.text and len(impression.text.strip()) > 20:
                    caption = impression.text.strip()
                elif mesh_terms:
                    caption = f"Chest X-ray showing {', '.join(mesh_terms)}."
                else:
                    caption = "Chest X-ray."

                if len(caption) > 512:
                    caption = caption[:512] + "..."
                
                for image in root.findall('.//parentImage'):
                    image_id = image.get('id')
                    if image_id:
                        image_path = os.path.join(self.image_dir, f"{image_id}.png")
                        if os.path.exists(image_path):
                            self.samples.append({
                                'image_path': image_path,
                                'caption': caption,
                                'report': report_text,
                                'mesh_terms': mesh_terms
                            })
            except Exception as e:
                logger.error(f"Error processing {xml_file}: {e}")
        
        logger.info(f"Loaded {len(self.samples)} samples")
        
        random.seed(42)
        random.shuffle(self.samples)
        
        split_idx = int(len(self.samples) * split_ratio)
        if is_train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.3), 
                transforms.RandomAffine(0, translate=(0.02, 0.02), scale=(0.98, 1.02)), 
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample['image_path']).convert('L')
        image = self.transform(image)
        
        return {
            'pixel_values': image,
            'caption': sample['caption'],
            'report': sample['report'],
            'mesh_terms': sample['mesh_terms']
        }


def custom_collate(batch):
    images = torch.stack([item['pixel_values'] for item in batch])
    captions = [item['caption'] for item in batch]
    reports = [item['report'] for item in batch]
    mesh_terms = [item['mesh_terms'] for item in batch]
    
    return {
        'pixel_values': images,
        'caption': captions,
        'report': reports,
        'mesh_terms': mesh_terms
    }
