import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
import json
import os


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class VQualADataset(Dataset):
    def __init__(self, type, module):
        self.data = []
        self.type = type
        if type == 'train':
            with open(f'/root/autodl-tmp/VQualA/key_frames/train/data_info.json', 'rt') as f:
                self.data = json.load(f)['videos']
        else:
            with open(f'/root/autodl-tmp/VQualA/key_frames/val/data_info.json', 'rt') as f:
                self.data = json.load(f)['videos']

        if module == "traditional_module":
            self.video_size = (384, 384)
        else:
            self.video_size = (224, 224)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        frame_path = item["frame_path"]
        num_frames = item["extracted_frames"]
        prompt = item["Prompt"]
        Overall_MOS = item["Overall_MOS"]
        Traditional_MOS = item["Traditional_MOS"]
        Alignment_MOS = item["Alignment_MOS"]
        Aesthetic_MOS = item["Aesthetic_MOS"]
        Temporal_MOS = item["Temporal_MOS"]

        transform_fn = transforms.Resize(self.video_size)
        normalize_fn = transforms.Normalize(mean, std)
        key_frames = []

        for i in range(num_frames):
            frame = cv2.imread(os.path.join(frame_path, '{:03d}'.format(i) + '.png'))
            frame = cv2.cvtColor(frame, 4) #4: BGR2RGB
            frame = Image.fromarray(frame, "RGB")
            frame = transforms.ToTensor()(transform_fn(frame)) #Resize -> (224, 224)
            frame = normalize_fn(frame)

            key_frames.append(frame)

        video = torch.stack(key_frames, 0)

        data = {
            "video": video, #B, N, C, H, W
            "prompt": prompt,
            "Overall_MOS": Overall_MOS,
            "Traditional_MOS": Traditional_MOS,
            "Alignment_MOS": Alignment_MOS,
            "Aesthetic_MOS": Aesthetic_MOS,
            "Temporal_MOS": Temporal_MOS
        }

        return data
    

def custom_collate_fn(batch):
    #处理不同帧数的视频
    max_frames = max([item["video"].shape[0] for item in batch])

    videos = []
    for item in batch:
        video = item["video"]
        if video.shape[0] < max_frames:
            #重复最后一帧来填充
            last_frame = video[-1, :, :, :].unsqueeze(0)
            repeat_times = max_frames - video.shape[0]
            repeat_frames = last_frame.repeat(repeat_times, 1, 1, 1)
            video = torch.cat([video, repeat_frames], dim=0)
        videos.append(video)

    return {
        "video": torch.stack(videos, 0),
        "prompt": [item["prompt"] for item in batch],
        "Overall_MOS": torch.tensor([item["Overall_MOS"] for item in batch]),
        "Traditional_MOS": torch.tensor([item["Traditional_MOS"] for item in batch]),
        "Alignment_MOS": torch.tensor([item["Alignment_MOS"] for item in batch]),
        "Aesthetic_MOS": torch.tensor([item["Aesthetic_MOS"] for item in batch]),
        "Temporal_MOS": torch.tensor([item["Temporal_MOS"] for item in batch])
    }

if __name__ == '__main__':
    dataset = VQualADataset('train')
    data = dataset.__getitem__(0)
    print(data["video"].shape)
