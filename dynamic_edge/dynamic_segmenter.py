import cv2
from typing import List
from query_analyzer import TemporalClass

class DynamicSegmenter:
    def __init__(self, base_M: int = 64):
        """
        Initialize the segmenter with the default M from the SEC paper.
        """
        self.base_M = base_M
        
    def determine_M(self, temporal_class: TemporalClass, video_length_sec: float) -> int:
        """
        Dynamically determine the number of segments M based on temporal class and video length.
        """
        if temporal_class == TemporalClass.STATIC_SCENE:
            # Static scenes need fewer segments. 
            # E.g., 1 frame every 5 seconds, minimum 10.
            return max(10, int(video_length_sec / 5.0))
            
        elif temporal_class == TemporalClass.FAST_ACTION:
            # Fast actions need dense sampling.
            # E.g., 3 frames per second, ensuring we don't miss quick events.
            return max(self.base_M, int(video_length_sec * 3.0))
            
        else: # LONG_DURATION
            # Use the base M or a steady sampling rate.
            return self.base_M

    def extract_candidate_frames(self, video_path: str, temporal_class: TemporalClass) -> List[dict]:
        """
        Dynamically divides the video into M segments and extracts the central frame of each.
        Returns a list of dictionaries with frame info.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_length_sec = total_frames / fps if fps > 0 else 0
        
        # Calculate dynamic M
        M = self.determine_M(temporal_class, video_length_sec)
        print(f"Video Length: {video_length_sec:.2f}s. Dynamic M determined as: {M}")
        
        frames = []
        if total_frames == 0 or M == 0:
            return frames
            
        # Size of each segment in frames
        segment_size = total_frames / M
        
        for i in range(M):
            # Central frame index of the i-th segment
            center_frame_idx = int((i + 0.5) * segment_size)
            if center_frame_idx >= total_frames:
                center_frame_idx = total_frames - 1
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, center_frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for standard processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append({
                    'frame_idx': center_frame_idx,
                    'timestamp_sec': center_frame_idx / fps,
                    'image': frame_rgb
                })
                
        cap.release()
        return frames

if __name__ == "__main__":
    # Small test
    segmenter = DynamicSegmenter()
    print("Testing M calculations for a 30s video:")
    print("STATIC_SCENE:", segmenter.determine_M(TemporalClass.STATIC_SCENE, 30.0))
    print("LONG_DURATION:", segmenter.determine_M(TemporalClass.LONG_DURATION, 30.0))
    print("FAST_ACTION:", segmenter.determine_M(TemporalClass.FAST_ACTION, 30.0))
