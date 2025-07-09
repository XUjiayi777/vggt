import cv2
import glob
import os

def video_2_images(video_path,output_folder):
    if os.path.isdir(video_path):
        img_paths = sorted(glob.glob(f"{video_path}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {video_path}")
        frame_interval = int(video_fps)
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Total frames: {total_frames}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_lists = []
        for i in range(len(frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[i])
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_lists.append(frame_path)
        cap.release()
    return img_lists

    