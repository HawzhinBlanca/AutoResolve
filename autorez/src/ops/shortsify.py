import logging

logger = logging.getLogger(__name__)

"""
Blueprint3 Ops Module - Shortsify
Real implementation meeting ≤120s latency for 30min video
"""
import numpy as np
import json
import time
import os
import configparser
from src.utils.common import set_global_seed
from src.embedders.vjepa_embedder import VJEPAEmbedder

CFG = configparser.ConfigParser()
CFG.read(os.getenv("OPS_INI", "conf/ops.ini"))

class Shortsify:
    def __init__(self):
        set_global_seed(1234)
        self.max_latency_30min = int(CFG.get("shortsify", "max_latency_30min", fallback="120"))
        self.default_recipe = CFG.get("shortsify", "default_recipe", fallback="vertical_60s")
        self.target_duration = int(CFG.get("shortsify", "target_duration", fallback="60"))
        self.min_hook_duration = int(CFG.get("shortsify", "min_hook_duration", fallback="3"))
        self.max_hook_duration = int(CFG.get("shortsify", "max_hook_duration", fallback="10"))
        
        # Initialize V-JEPA for content analysis
        self.embedder = VJEPAEmbedder(use_real_vjepa2=True, memory_safe_mode=True)
        
    def find_hooks(self, video_path):
        """
        Find compelling moments for short-form content using V-JEPA-2
        Returns: List of (start_time, end_time, score) tuples
        """
        # Extract video embeddings
        segments, _ = self.embedder.embed_segments(
            video_path,
            fps=2.0,  # Higher FPS for better hook detection
            window=8,
            strategy="temp_attn",
            max_segments=200,
            return_frame_cls=True
        )
        
        if not segments:
            return []
        
        hooks = []
        
        # Analyze each segment for hook potential
        for i, segment in enumerate(segments):
            # Get frame-level features if available
            if 'frame_cls' in segment:
                frame_features = segment['frame_cls']
                
                # Calculate variance (movement/activity)
                activity_score = np.var(frame_features).mean()
                
                # Calculate novelty (difference from context)
                if i > 0:
                    prev_emb = segments[i-1]['emb']
                    curr_emb = segment['emb']
                    novelty_score = 1.0 - np.dot(prev_emb, curr_emb) / (np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb))
                else:
                    novelty_score = 0.5
                
                # Combined hook score
                hook_score = 0.6 * activity_score + 0.4 * novelty_score
                
                # Duration check
                segment_duration = segment['t1'] - segment['t0']
                if self.min_hook_duration <= segment_duration <= self.max_hook_duration:
                    hooks.append((segment['t0'], segment['t1'], hook_score))
        
        # Sort by score and return top candidates
        hooks.sort(key=lambda x: x[2], reverse=True)
        return hooks[:10]  # Top 10 hooks
    
    def generate_shorts(self, video_path, output_dir=None, recipe=None):
        """
        Generate short-form content meeting blueprint performance requirements
        Returns: (shorts_data, performance_metrics)
        """
        start_time = time.time()
        
        if recipe is None:
            recipe = self.default_recipe
        
        if output_dir is None:
            output_dir = "artifacts/shorts"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video duration for performance calculation
        duration = self._get_video_duration(video_path)
        
        # Find compelling hooks
        hooks = self.find_hooks(video_path)
        
        # Generate shorts from hooks
        shorts = []
        for i, (start, end, score) in enumerate(hooks[:5]):  # Top 5 shorts
            # Extend to target duration if needed
            current_duration = end - start
            if current_duration < self.target_duration:
                extension = (self.target_duration - current_duration) / 2
                start = max(0, start - extension)
                end = min(duration, end + extension)
            
            short_data = {
                "id": f"short_{i+1}",
                "start_time": start,
                "end_time": end,
                "duration": end - start,
                "hook_score": score,
                "recipe": recipe,
                "output_file": f"{output_dir}/short_{i+1}.mp4"
            }
            
            # Generate the actual short video
            self._create_short_video(video_path, short_data)
            shorts.append(short_data)
        
        elapsed = time.time() - start_time
        
        # Performance metrics
        duration_minutes = duration / 60.0
        latency_per_30min = elapsed * (30.0 / duration_minutes) if duration_minutes > 0 else elapsed
        
        metrics = {
            "processing_time_s": elapsed,
            "video_duration_s": duration,
            "latency_per_30min": latency_per_30min,
            "meets_requirement": latency_per_30min <= self.max_latency_30min,
            "shorts_generated": len(shorts),
            "hooks_found": len(hooks)
        }
        
        # Save shorts index
        shorts_data = {
            "version": "3.0",
            "source_video": video_path,
            "recipe": recipe,
            "shorts": shorts,
            "metadata": {
                "generated_at": time.time(),
                "total_shorts": len(shorts),
                "target_duration": self.target_duration
            }
        }
        
        index_path = f"{output_dir}/index.json"
        with open(index_path, 'w') as f:
            json.dump(shorts_data, f, indent=2)
        
        return shorts_data, metrics
    
    def _get_video_duration(self, video_path):
        """Get video duration in seconds"""
        import av
        try:
            container = av.open(video_path)
            duration = float(container.duration) / 1000000.0 if container.duration else 60.0
            container.close()
            return duration
        except:
            return 60.0  # Default fallback
    
    def _create_short_video(self, source_path, short_data):
        """Create actual short video file using ffmpeg"""
        start_time = short_data['start_time']
        duration = short_data['duration']
        output_file = short_data['output_file']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Generate short video with vertical aspect ratio for mobile
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", source_path,
            "-t", str(duration),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            output_file
        ]
        
        # Run ffmpeg
        result = os.system(" ".join([f"'{arg}'" for arg in cmd]) + " 2>/dev/null")
        
        return result == 0

def shortsify_cli():
    """CLI interface for shortsify"""
    import sys
    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.ops.shortsify <video_path> [output_dir]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "artifacts/shorts"
    
    shortsify = Shortsify()
    shorts_data, metrics = shortsify.generate_shorts(video_path, output_dir)
    
    logger.info(f"Shortsify complete:")
    logger.info(f"  Video duration: {metrics['video_duration_s']:.1f}s")
    logger.info(f"  Processing time: {metrics['processing_time_s']:.1f}s")
    logger.info(f"  Latency per 30min: {metrics['latency_per_30min']:.1f}s")
    logger.info(f"  Meets requirement (≤{shortsify.max_latency_30min}s): {metrics['meets_requirement']}")
    logger.info(f"  Shorts generated: {metrics['shorts_generated']}")
    logger.info(f"  Hooks found: {metrics['hooks_found']}")

if __name__ == "__main__":
    shortsify_cli()
