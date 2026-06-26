import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict

try:
    from transformers import ClapModel, ClapProcessor
except ImportError:
    ClapModel, ClapProcessor = None, None

class SemanticFrameSelector:
    def __init__(self, model_name="openai/clip-vit-base-patch32", alpha=0.8, gamma=0.8, audio_model_name="laion/clap-htsat-unfused"):
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
        
        self.clap_model = None
        self.clap_processor = None
        if ClapModel is not None:
            try:
                print(f"Loading CLAP model {audio_model_name} on {self.device}...")
                self.clap_model = ClapModel.from_pretrained(audio_model_name).to(self.device)
                self.clap_processor = ClapProcessor.from_pretrained(audio_model_name)
            except Exception as e:
                print(f"Could not load CLAP model: {e}")

    def extract_features(self, frames: List[Dict], query: str, batch_size=16):
        images = [Image.fromarray(f['image']) for f in frames]
        
        all_image_embeds = []
        text_embeds = None
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            inputs = self.processor(text=[query], images=batch_images, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                all_image_embeds.append(image_embeds)
                
                if text_embeds is None:
                    text_embeds = outputs.text_embeds
                    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
        final_image_embeds = torch.cat(all_image_embeds, dim=0)
        return final_image_embeds, text_embeds

    def compute_similarity(self, image_embeds, text_embeds):
        sim = torch.matmul(text_embeds, image_embeds.t()).squeeze(0)
        return sim.cpu().numpy()

    def extract_audio_features(self, frames: List[Dict], query: str, batch_size=16):
        if self.clap_model is None or self.clap_processor is None:
            return None, None
            
        has_audio = any(f.get('audio') is not None for f in frames)
        if not has_audio:
            return None, None
            
        valid_audio = []
        for f in frames:
            if f.get('audio') is not None:
                valid_audio.append(f['audio'])
            else:
                valid_audio.append(np.zeros(48000 * 2)) # 2s of silence
                
        all_audio_embeds = []
        text_embeds = None
        
        for i in range(0, len(valid_audio), batch_size):
            batch_audios = valid_audio[i:i+batch_size]
            inputs = self.clap_processor(text=[query], audios=batch_audios, return_tensors="pt", padding=True, sampling_rate=48000).to(self.device)
            
            with torch.no_grad():
                outputs = self.clap_model(**inputs)
                audio_embeds = outputs.audio_embeds
                audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
                all_audio_embeds.append(audio_embeds)
                
                if text_embeds is None:
                    text_embeds = outputs.text_embeds
                    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
        final_audio_embeds = torch.cat(all_audio_embeds, dim=0)
        return final_audio_embeds, text_embeds

    def compute_audio_similarity(self, audio_embeds, text_embeds):
        if audio_embeds is None:
            return None
        sim = torch.matmul(text_embeds, audio_embeds.t()).squeeze(0)
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
        visual_sim_scores = self.compute_similarity(image_embeds, text_embeds)
        
        audio_embeds, audio_text_embeds = self.extract_audio_features(candidate_frames, query)
        audio_sim_scores = self.compute_audio_similarity(audio_embeds, audio_text_embeds)
        
        sim_scores = visual_sim_scores.copy()
        if audio_sim_scores is not None:
            print("Fusing visual and audio similarity scores!")
            beta = 0.5 # Equal weighting for vision and audio
            sim_scores = beta * visual_sim_scores + (1 - beta) * audio_sim_scores
        
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
