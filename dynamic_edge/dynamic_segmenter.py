import cv2
import numpy as np
from typing import List
from query_analyzer import TemporalClass

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    VideoFileClip = None

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
        Also extracts a 2-second audio chunk around the frame if audio exists.
        Returns a list of dictionaries with frame and audio info.
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
        
        # Check for audio
        has_audio = False
        video_clip = None
        if VideoFileClip is not None:
            try:
                video_clip = VideoFileClip(video_path)
                if video_clip.audio is not None:
                    has_audio = True
                    print(f"Audio track found! Will extract audio chunks.")
            except Exception as e:
                print(f"Note: Could not load audio track (maybe there isn't one). {e}")

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
                center_timestamp_sec = center_frame_idx / fps
                
                # Extract audio chunk (+/- 1 second)
                audio_array = None
                if has_audio and video_clip.audio is not None:
                    start_t = max(0, center_timestamp_sec - 1.0)
                    end_t = min(video_clip.duration, center_timestamp_sec + 1.0)
                    if start_t < end_t:
                        try:
                            audio_subclip = video_clip.audio.subclip(start_t, end_t)
                            # CLAP uses 48kHz by default
                            audio_array = audio_subclip.to_soundarray(fps=48000)
                            # Convert to mono if it's stereo
                            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                                audio_array = audio_array.mean(axis=1)
                        except Exception as e:
                            # Silently fail for audio chunk errors to not spam
                            pass

                frames.append({
                    'frame_idx': center_frame_idx,
                    'timestamp_sec': center_timestamp_sec,
                    'image': frame_rgb,
                    'audio': audio_array
                })
                
        cap.release()
        if video_clip is not None:
            video_clip.close()
        return frames

if __name__ == "__main__":
    # Small test
    segmenter = DynamicSegmenter()
    print("Testing M calculations for a 30s video:")
    print("STATIC_SCENE:", segmenter.determine_M(TemporalClass.STATIC_SCENE, 30.0))
    print("LONG_DURATION:", segmenter.determine_M(TemporalClass.LONG_DURATION, 30.0))
    print("FAST_ACTION:", segmenter.determine_M(TemporalClass.FAST_ACTION, 30.0))
