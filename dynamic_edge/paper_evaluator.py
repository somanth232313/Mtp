import os
import time
import torch
import csv
import psutil
import urllib.request
import cv2
import numpy as np
from query_analyzer import QueryAnalyzer
from dynamic_segmenter import DynamicSegmenter
from edge_frame_selection import SemanticFrameSelector
from PIL import Image

def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
    else:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

def generate_needle_in_haystack_video(path):
    print(f"Generating sparse action video: {path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (224, 224))
    
    # 15 seconds total (450 frames at 30fps)
    # Needle is at frames 300 to 380
    
    for i in range(450):
        # Default: empty dark room
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Action: bright white circle
        if 300 <= i <= 380:
            cv2.circle(frame, (112, 112), 50, (255, 255, 255), -1)
            
        out.write(frame)
    out.release()

class UniformSamplingBaseline:
    """Naive baseline that just takes N evenly spaced frames without AI models."""
    def extract(self, video_path, N):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []
            
        segment_size = total_frames / N
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        for i in range(N):
            idx = int((i + 0.5) * segment_size)
            if idx >= total_frames: 
                idx = total_frames - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append({
                    'frame_idx': idx,
                    'timestamp_sec': idx / fps,
                    'image': frame_rgb,
                    'audio': None 
                })
        cap.release()
        return frames

def calculate_peak_relevance_score(selector, frames, query):
    """Uses the SemanticFrameSelector to grade the MAX relevance (Top-1) of the selected frames."""
    if not frames:
        return 0.0
    image_embeds, text_embeds = selector.extract_features(frames, query, batch_size=16)
    sim_scores = selector.compute_similarity(image_embeds, text_embeds)
    # Return max visual semantic score (Did we find the needle?)
    return float(np.max(sim_scores))

def evaluate():
    print("="*60)
    print("Ablation Study: Dynamic Edge Pipeline vs Naive Baseline")
    print("="*60)
    
    # The Needle in a Haystack test case
    test_cases = [
        {"video": "sparse_action_test.mp4", "query": "a bright white circle", "N": 4}
    ]
    
    for tc in test_cases:
        generate_needle_in_haystack_video(tc["video"])

    print("\nLoading AI models for evaluation... (This may take a moment)")
    analyzer = QueryAnalyzer()
    segmenter = DynamicSegmenter(base_M=32)
    # Use low gamma to prioritize pure accuracy over visual diversity
    selector = SemanticFrameSelector(alpha=0.9, gamma=0.1)
    baseline = UniformSamplingBaseline()

    results = []

    for tc in test_cases:
        video_path = tc["video"]
        query = tc["query"]
        N = tc["N"]
        
        # --- Run Baseline ---
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        base_frames = baseline.extract(video_path, N)
        base_latency = time.time() - start_time
        base_peak_mem = get_memory_usage()
        # Evaluate Top-1 score
        base_score = calculate_peak_relevance_score(selector, base_frames, query)
        
        results.append({
            "Method": "Uniform Baseline",
            "Video": video_path,
            "Query": query,
            "Latency_s": round(base_latency, 2),
            "Peak_Relevance_Score": round(base_score, 4),
            "Peak_VRAM_MB": round(base_peak_mem, 2)
        })

        # --- Run Dynamic Edge ---
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        t_class = analyzer.analyze(query)
        cand_frames = segmenter.extract_candidate_frames(video_path, t_class)
        final_frames = selector.select_frames(cand_frames, query, N)
        
        edge_latency = time.time() - start_time
        edge_peak_mem = get_memory_usage()
        # Evaluate Top-1 score
        edge_score = calculate_peak_relevance_score(selector, final_frames, query)
        
        results.append({
            "Method": "Dynamic Edge",
            "Video": video_path,
            "Query": query,
            "Latency_s": round(edge_latency, 2),
            "Peak_Relevance_Score": round(edge_score, 4),
            "Peak_VRAM_MB": round(edge_peak_mem, 2)
        })

    # Print Markdown Table
    print("\n### Final Ablation Study Results (Needle in a Haystack Test)\n")
    print("| Method | Video | Query | Latency (s) | Peak Semantic Score | Peak VRAM (MB) |")
    print("|--------|-------|-------|-------------|---------------------|----------------|")
    for r in results:
        print(f"| {r['Method']} | {r['Video']} | {r['Query']} | {r['Latency_s']} | {r['Peak_Relevance_Score']} | {r['Peak_VRAM_MB']} |")
        
    csv_file = "success_results.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved detailed comparative results to {csv_file}")

if __name__ == "__main__":
    evaluate()
