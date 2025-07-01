import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision import transforms
import cv2
from PIL import Image
import pandas as pd
import os


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class BaseVQualADataset(Dataset):
    """基础数据集类"""

    def __init__(self,
                 dataset_type: str,
                 num_splits: int,
                 video_size: int = 224,
                 num_frames: int = 16,
                 video_clip_min: int = 8,
                 video_clip_length: int = 32):
        super().__init__()

        self.dataset_type = dataset_type
        self.video_size = video_size
        self.num_frames = num_frames
        self.video_clip_min = video_clip_min
        self.video_clip_length = video_clip_length
        self.is_test = dataset_type == 'test'

        if dataset_type == 'train':
            self.csv_path = f"/root/autodl-tmp/VQualA/data/train_{num_splits+1}.csv"
            self.videos_dir = "/root/autodl-tmp/VQualA/data/train"
        elif dataset_type == 'val':
            self.csv_path = f"/root/autodl-tmp/VQualA/data/val_{num_splits+1}.csv"
            self.videos_dir = "/root/autodl-tmp/VQualA/data/train"
        else:
            self.csv_path = "/root/autodl-tmp/VQualA/data/test.csv"
            self.videos_dir = "/root/autodl-tmp/VQualA/data/test"

        self._load_data()

        self.transform = transforms.Compose([
            transforms.Resize((self.video_size, self.video_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def _load_data(self):
        try:
            dataInfo = pd.read_csv(self.csv_path, encoding='utf-8')

            self.video_names = dataInfo["video_name"].tolist()
            self.prompts = dataInfo["Prompt"].tolist()

            if self.is_test:
                num_samples = len(self.video_names)
                default_value = 0.0
                self.overall_mos = [default_value] * num_samples
                self.traditional_mos = [default_value] * num_samples
                self.alignment_mos = [default_value] * num_samples
                self.aesthetic_mos = [default_value] * num_samples
                self.temporal_mos = [default_value] * num_samples
            else:
                self.overall_mos = dataInfo["Overall_MOS"].tolist()
                self.traditional_mos = dataInfo["Traditional_MOS"].tolist()
                self.alignment_mos = dataInfo["Alignment_MOS"].tolist()
                self.aesthetic_mos = dataInfo["Aesthetic_MOS"].tolist()
                self.temporal_mos = dataInfo["Temporal_MOS"].tolist()

        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise

    def __len__(self):
        return len(self.video_names)
    
    def extract_frames(self, video_name: str) -> list:
        filename = os.path.join(self.videos_dir, video_name)
        
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print(f"Error: Could not open video {filename}")
            return None
        
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        
        extracted_frames = []
        video_read_index = 0
        frame_idx = 0

        if video_frame_rate != 0:
            for i in range(video_length):
                has_frames, frame = cap.read()
                if has_frames:
                    if frame_idx % int(video_frame_rate / 2) == 0:
                        extracted_frames.append(frame.copy())
                        video_read_index += 1
                    frame_idx += 1
        else:
            for i in range(video_length):
                has_frames, frame = cap.read()
                if has_frames:
                    if video_read_index < video_length:
                        extracted_frames.append(frame.copy())
                        video_read_index += 1
        
        cap.release()

        if len(extracted_frames) == 0:
            print(f"警告: 视频 {video_name} 没有提取到任何帧")
            return None
        
        #处理不足或超出的帧数
        if len(extracted_frames) < self.num_frames:
            last_frame = extracted_frames[-1]
            for _ in range(self.num_frames - len(extracted_frames)):
                extracted_frames.append(last_frame.copy())
        elif len(extracted_frames) > self.num_frames:
            extracted_frames = extracted_frames[:self.num_frames]
        
        return extracted_frames
    
    def process_frames_to_tensor(self, frames: list) -> Tensor:
        """将提取的帧列表转换为张量"""
        processed_frames = []

        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame, "RGB")
            frame_tensor = self.transform(frame)
            processed_frames.append(frame_tensor)

        return torch.stack(processed_frames, 0) #[N, C, H, W]
    
    def extract_video_clips(self, video_name: str) -> Tensor:
        """提取视频片段——原版"""
        video_path = os.path.join(self.videos_dir, video_name)
        
        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
        
        if frame_rate == 0:
            video_clip = 10
        else:
            video_clip = int(video_length / frame_rate)
        
        #读取所有帧
        transformed_frame_all = torch.zeros([video_length, 3, self.video_size, self.video_size])
        video_read_index = 0

        for i in range(video_length):
            has_frame, frame = cap.read()
            if has_frame:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = self.transform(frame)
                transformed_frame_all[video_read_index] = frame
                video_read_index += 1
        
        #填充缺失帧
        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
        
        cap.release()

        #创建视频片段
        transformed_video_all = []
        for i in range(video_clip):
            transformed_video = torch.zeros([self.video_clip_length, 3, self.video_size, self.video_size])
            if (i * frame_rate + self.video_clip_length) <= video_length:
                transformed_video = transformed_frame_all[i * frame_rate : (i * frame_rate + self.video_clip_length)]
            else:
                transformed_video[:(video_length - i * frame_rate)] = transformed_frame_all[i * frame_rate :]
                for j in range((video_length - i * frame_rate), self.video_clip_length):
                    transformed_video[j] = transformed_video[video_length - i * frame_rate - 1]
            transformed_video_all.append(transformed_video)
        
        #确保最小片段数
        if video_clip < self.video_clip_min:
            for i in range(video_clip, self.video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
        
        return torch.stack(transformed_video_all, dim=0)  # [M, N, C, H, W]
    

class VQualADataset(BaseVQualADataset):
    """通用数据集类, 根据模式返回不同数据"""

    def __init__(self,
                 dataset_type: str,
                 mode: str = 'all',
                 traditional_size: int = 384,
                 **kwargs):
        super().__init__(dataset_type, **kwargs)
        self.mode = mode
        self.traditional_size = traditional_size

        if mode not in ['image', 'video', 'all']:
            raise ValueError("mode must be one of : 'image', 'video', 'all'")
        
        self.traditional_transform = transforms.Compose([
            transforms.Resize((self.traditional_size, self.traditional_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def process_frames_to_tensor_traditional(self, frames: list) -> Tensor:
        """将提取的帧列表转换为传统模块所需的张量 (384x384)"""
        processed_frames = []

        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame, "RGB")
            frame_tensor = self.traditional_transform(frame)
            processed_frames.append(frame_tensor)

        return torch.stack(processed_frames, 0) #[N, C, H, W]
        
        
    def __getitem__(self, idx: int) -> dict:
        video_name = self.video_names[idx]

        data = {
            "prompt": self.prompts[idx],
            "video_name": video_name
        }
        
        if not self.is_test:
            data.update({
                "Overall_MOS": self.overall_mos[idx],
                "Traditional_MOS": self.traditional_mos[idx],
                "Alignment_MOS": self.alignment_mos[idx],
                "Aesthetic_MOS": self.aesthetic_mos[idx],
                "Temporal_MOS": self.temporal_mos[idx]
            })

        if self.mode == 'image':
            frames = self.extract_frames(video_name)
            if frames is not None and self.video_size == 224:
                data["image"] = self.process_frames_to_tensor(frames)
            elif frames is not None and self.video_size == 384:
                data["image"] = self.process_frames_to_tensor_traditional(frames)

        elif self.mode == 'video':
            data["video"] = self.extract_video_clips(video_name)

        elif self.mode == 'all':
            frames = self.extract_frames(video_name)

            data["image"] = self.process_frames_to_tensor(frames)
            data["image_traditional"] = self.process_frames_to_tensor_traditional(frames)
            data["video"] = self.extract_video_clips(video_name)

        return data
        

class DatasetImage(VQualADataset):
    """图像数据集"""
    def __init__(self, dataset_type: str, num_splits: int, num_frames: int = 6, video_size: int = 224, traditional_size: int = 384):
        super().__init__(dataset_type, num_splits=num_splits, mode='image', num_frames=num_frames, video_size=video_size, traditional_size=traditional_size)


class DatasetVideo(VQualADataset):
    """视频数据集"""
    def __init__(self, dataset_type: str, num_splits: int, video_size: int = 224, video_clip_min: int = 8, video_clip_length: int = 32):
        super().__init__(dataset_type, num_splits=num_splits, mode='video', video_size=video_size, video_clip_min=video_clip_min, video_clip_length=video_clip_length)


class DatasetTrack1(VQualADataset):
    """混合数据集"""
    def __init__(self, dataset_type: str, num_splits: int, num_frames: int = 6, video_size: int = 224, traditional_size: int = 384, video_clip_min: int = 8, video_clip_length: int = 32):
        super().__init__(dataset_type, num_splits=num_splits, mode='all', num_frames=num_frames, video_size=video_size, traditional_size=traditional_size,
                        video_clip_min=video_clip_min, video_clip_length=video_clip_length)
        



