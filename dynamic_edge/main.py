from query_analyzer import QueryAnalyzer
from dynamic_segmenter import DynamicSegmenter
from edge_frame_selection import SemanticFrameSelector
import time

def run_pipeline(video_path: str, query: str, N: int = 8):
    print(f"\n{'='*50}")
    print(f"Starting Dynamic Edge Pipeline")
    print(f"Query: '{query}'")
    print(f"{'='*50}")
    start_time = time.time()
    
    # 1. Analyze Query (Our Novel Contribution)
    analyzer = QueryAnalyzer()
    temporal_class = analyzer.analyze(query)
    print(f"\n[Step 1] Query Analyzer Result -> {temporal_class.name}")
    
    # 2. Dynamic Segmentation (Adaptive to the Query)
    segmenter = DynamicSegmenter(base_M=64)
    print(f"[Step 2] Extracting dynamic candidate frames...")
    try:
        candidate_frames = segmenter.extract_candidate_frames(video_path, temporal_class)
        print(f"         Extracted {len(candidate_frames)} candidate frames.")
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return
    
    # 3. Semantic Frame Selection (SEC Paper's Core Math)
    if len(candidate_frames) > 0:
        selector = SemanticFrameSelector(alpha=0.8, gamma=0.8)
        print(f"\n[Step 3] Selecting top {N} frames using Semantic Match + Marginal Relevance...")
        final_frames = selector.select_frames(candidate_frames, query, N)
        
        print(f"\n[Finished] Total Time: {time.time() - start_time:.2f}s")
        print(f"Final Selected Frame Timestamps (sec): {[f'{f['timestamp_sec']:.2f}' for f in final_frames]}")
    else:
        print("No frames extracted.")
    
if __name__ == "__main__":
    print("Welcome to the Query-Aware Dynamic Edge Selection Pipeline!")
    print("To test, please modify this file to point to a valid .mp4 video on your desktop.")
    
    # Example usage:
    # run_pipeline("c:/Users/hp/Desktop/MTP/sample_action.mp4", "Did the person drop the bag?", N=8)
    # run_pipeline("c:/Users/hp/Desktop/MTP/sample_static.mp4", "What color is the sky?", N=8)
