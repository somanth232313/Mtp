import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict

class SemanticFrameSelector:
    def __init__(self, model_name="openai/clip-vit-base-patch32", alpha=0.8, gamma=0.8):
        """
        Initializes the Semantic Frame Selector using a lightweight CLIP model.
        alpha: trades off temporal vs visual uniqueness in Marginal Relevance
        gamma: trades off Semantic Match vs Marginal Relevance
        """
        self.alpha = alpha
        self.gamma = gamma
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def extract_features(self, frames: List[Dict], query: str):
        images = [Image.fromarray(f['image']) for f in frames]
        inputs = self.processor(text=[query], images=images, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # Normalize
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        return image_embeds, text_embeds

    def compute_similarity(self, image_embeds, text_embeds):
        sim = torch.matmul(text_embeds, image_embeds.t()).squeeze(0)
        return sim.cpu().numpy()

    def temporal_distance(self, candidate_idx: int, selected_indices: List[int], total_frames: int):
        if not selected_indices:
            return 1.0 # Max distance if nothing is selected yet
        
        min_dist = min([abs(candidate_idx - s_idx) for s_idx in selected_indices])
        return min_dist / total_frames

    def visual_uniqueness(self, candidate_embed, selected_embeds):
        if len(selected_embeds) == 0:
            return 1.0
            
        sims = torch.matmul(torch.stack(selected_embeds), candidate_embed)
        max_sim = torch.max(sims).item()
        return 1.0 - max_sim

    def select_frames(self, candidate_frames: List[Dict], query: str, N: int) -> List[Dict]:
        """
        Selects the top N frames using unified ranking (Semantic + Marginal Relevance).
        """
        if len(candidate_frames) <= N:
            return candidate_frames
            
        M = len(candidate_frames)
        total_frames_in_video = candidate_frames[-1]['frame_idx'] + 1 if M > 0 else 1
        
        image_embeds, text_embeds = self.extract_features(candidate_frames, query)
        sim_scores = self.compute_similarity(image_embeds, text_embeds)
        
        selected_frames = []
        selected_indices = []
        selected_embeds = []
        
        unselected_indices = list(range(M))
        
        for _ in range(N):
            best_idx = -1
            best_score = -float('inf')
            
            for idx in unselected_indices:
                frame_info = candidate_frames[idx]
                sim = sim_scores[idx]
                
                # Marginal Relevance Calculation
                d_t = self.temporal_distance(frame_info['frame_idx'], 
                                             [candidate_frames[i]['frame_idx'] for i in selected_indices], 
                                             total_frames_in_video)
                d_v = self.visual_uniqueness(image_embeds[idx], selected_embeds)
                
                mrs = self.alpha * d_t + (1 - self.alpha) * d_v
                
                # Unified Score Calculation
                score = self.gamma * sim + (1 - self.gamma) * mrs
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    
            # Add the winning frame to our selected pool
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
            selected_embeds.append(image_embeds[best_idx])
            selected_frames.append(candidate_frames[best_idx])
            
        # Return frames sorted chronologically
        selected_frames.sort(key=lambda x: x['frame_idx'])
        return selected_frames
