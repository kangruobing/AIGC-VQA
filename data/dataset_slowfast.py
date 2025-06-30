import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
import os
import pandas as pd


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Dataset_slowfast(Dataset):
    def __init__(self,
                 video_dir="/root/autodl-tmp/VQualA/data/train",
                 filepath="/root/autodl-tmp/VQualA/data/train.csv",
                 resize=224):
        super().__init__()

        dataInfo = pd.read_csv(filepath)
        video = dataInfo['video_name'].tolist()
        score = dataInfo['Temporal_MOS'].tolist()
        
        video_names = []
        for i in video:
            video_names.append(i)
        self.video_names = video_names
        self.score = score

        self.video_resize = resize

        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.video_dir = video_dir

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        score = self.score[idx]
        video_name = self.video_names[idx]

        filename = os.path.join(self.video_dir, video_name)

        cap = cv2.VideoCapture(filename)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #视频帧数
        frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS))) #视频帧率

        if frame_rate == 0:
            video_clip = 10
        else:
            video_clip = int(video_length / frame_rate) #num of video clip

        video_clip_min = 8
        video_clip_length = 32

        transformed_frame_all = torch.zeros([video_length, 3, self.video_resize, self.video_resize])
        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = cap.read()
            if has_frames:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = self.transform(frame) #Image -> Tensor
                transformed_frame_all[video_read_index] = frame
                video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

        cap.release()

        for i in range(video_clip):
            #[32, 3, 224, 224]
            transformed_video = torch.zeros([video_clip_length, 3, self.video_resize, self.video_resize])
            if (i*frame_rate + video_clip_length) <= video_length:
                transformed_video = transformed_frame_all[i*frame_rate : (i*frame_rate + video_clip_length)]
            else:
                transformed_video[:(video_length - i*frame_rate)] = transformed_frame_all[i*frame_rate :]
                for j in range((video_length - i*frame_rate), video_clip_length):
                    transformed_video[j] = transformed_video[video_length - i*frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        videos = torch.stack(transformed_video_all, dim=0) #[8, 32, 3, 224, 224] 

        data = {
            "video": videos,
            "Temporal_MOS": score
        }
       
        return data