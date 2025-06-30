import os
import cv2
import pandas as pd
import argparse
import json


def extract_frame(videos_dir: str, video_name: str, save_folder: str, target_frames: int = 6) -> dict:
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]

    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print(f"Error: Could not open video {filename}")
        return None

    # extract video's info
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"Video: {video_name}, Size: {video_height}x{video_width}, FPS: {video_frame_rate}, Frames: {video_length}")

    video_info = {
        "video_name": video_name,
        "video_path": filename,
        "total_frames": video_length,
        "frame_rate": video_frame_rate,
        "width": video_width,
        "height": video_height,
    }

    video_read_index = 0
    frame_idx = 0
    frame_path = exit_folder(os.path.join(save_folder, video_name_str))
    extracted_frames = []  # 存储提取的帧
    
    if video_frame_rate != 0:
        for i in range(video_length):
            has_frames, frame = cap.read()
            if has_frames:
                # key frame
                if frame_idx % video_frame_rate == 0:
                    extracted_frames.append(frame.copy())
                    video_read_index += 1
                frame_idx += 1
    else:  # to avoid the situation that the frame rate is less than 1 fps
        for i in range(video_length):
            has_frames, frame = cap.read()
            if has_frames:
                if video_read_index < video_length:
                    extracted_frames.append(frame.copy())
                    video_read_index += 1

    cap.release()

    # 处理帧数阈值
    if len(extracted_frames) == 0:
        print(f"警告: 视频 {video_name} 没有提取到任何帧")
        return None
    
    # 如果帧数少于目标帧数，用最后一帧填充
    if len(extracted_frames) < target_frames:
        last_frame = extracted_frames[-1]  # 获取最后一帧
        padding_count = target_frames - len(extracted_frames)
        print(f"视频 {video_name}: 原始帧数 {len(extracted_frames)}, 用最后一帧填充 {padding_count} 帧到 {target_frames} 帧")
        
        # 用最后一帧填充
        for _ in range(padding_count):
            extracted_frames.append(last_frame.copy())
    
    # 如果帧数超过目标帧数，截取前N帧
    elif len(extracted_frames) > target_frames:
        print(f"视频 {video_name}: 原始帧数 {len(extracted_frames)}, 截取前 {target_frames} 帧")
        extracted_frames = extracted_frames[:target_frames]

    # 保存所有帧
    for idx, frame in enumerate(extracted_frames):
        frame_filename = os.path.join(frame_path, '{:03d}.png'.format(idx))
        cv2.imwrite(frame_filename, frame)

    video_info["frame_path"] = frame_path
    video_info["extracted_frames"] = len(extracted_frames)

    print(f"视频 {video_name} 最终帧数: {len(extracted_frames)}")

    return video_info


def exit_folder(folder_name: str) -> str:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return folder_name


def main(config):
    videos_dir = config.videos_dir
    filename_path = config.filename_path
    save_folder = config.save_folder
    target_frames = config.target_frames

    # 读取数据集信息
    try:
        if filename_path.endswith('.xlsx') or filename_path.endswith('.xls'):
            dataInfo = pd.read_excel(filename_path)
        else:
            dataInfo = pd.read_csv(filename_path, encoding="utf-8")

        print(f"数据集列名: {dataInfo.columns.tolist()}")
        print(f"数据集形状: {dataInfo.shape}")

    except Exception as e:
        print(f"读取数据集文件失败: {e}")
        return
    
    all_video_info = []  # 存储所有信息的列表
    successful_count = 0
    failed_count = 0

    for idx, row in dataInfo.iterrows():
        video_name = row['video_name']
        print(f"\n开始处理第 {idx+1}/{len(dataInfo)} 个视频: {video_name}")

        video_info = extract_frame(videos_dir, video_name, save_folder, target_frames)

        if video_info is not None:
            video_info.update({
                "Prompt": row.get('Prompt', None),
                "Overall_MOS": row.get('Overall_MOS', None),
                "Traditional_MOS": row.get('Traditional_MOS', None),
                "Alignment_MOS": row.get('Alignment_MOS', None),
                "Aesthetic_MOS": row.get('Aesthetic_MOS', None),
                "Temporal_MOS": row.get("Temporal_MOS", None)
            })

            all_video_info.append(video_info)
            successful_count += 1
        else:
            failed_count += 1
            print(f"处理视频 {video_name} 失败")

    # 创建json文件
    json_path = os.path.join(config.json_path, 'data_info.json')

    output_data = {
        "dataset_info": {  
            "total_videos": len(all_video_info),
            "target_frames": target_frames,
            "dataset_path": filename_path,
            "videos_directory": videos_dir,
            "frames_directory": save_folder
        },
        "videos": all_video_info
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 处理完成 ===")
    print(f"视频信息已保存到: {json_path}")
    print(f"成功处理: {successful_count} 个视频")
    print(f"处理失败: {failed_count} 个视频")
    print(f"所有视频统一为 {target_frames} 帧")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--videos_dir', default='/root/autodl-tmp/VQualA/data/val', type=str, help='视频文件目录')
    parser.add_argument('--filename_path', default='/root/autodl-tmp/VQualA/data/val.csv', type=str, help='数据集CSV文件路径')
    parser.add_argument('--save_folder', default='/root/autodl-tmp/VQualA/key_frames/val/data', type=str, help='保存帧的目录')
    parser.add_argument('--json_path', default='/root/autodl-tmp/VQualA/key_frames/val', type=str, help='保存JSON文件的目录')
    parser.add_argument('--target_frames', default=6, type=int, help='目标帧数, 默认为6')

    config = parser.parse_args()

    main(config)