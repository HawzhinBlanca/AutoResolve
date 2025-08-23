"""
Determinism Validation Framework for AutoResolve v3.0
Ensures all components produce identical outputs for identical inputs
"""

import hashlib
import json
import numpy as np
import torch
import random
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeterminismTest:
    """Result of a determinism test"""
    component: str
    input_hash: str
    output_hash_run1: str
    output_hash_run2: str
    is_deterministic: bool
    execution_time_run1: float
    execution_time_run2: float
    details: Dict[str, Any]

class DeterminismValidator:
    """
    Validates that all AutoResolve components are deterministic.
    Tests each component multiple times with same input to ensure identical outputs.
    """
    
    def __init__(self, seed: int = 1234):
        self.seed = seed
        self.test_results: List[DeterminismTest] = []
        
    def set_all_seeds(self, seed: Optional[int] = None):
        """Set all random seeds for deterministic execution"""
        if seed is None:
            seed = self.seed
            
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        # PyTorch deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Environment variables for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # MPS (Metal Performance Shaders) determinism for Mac
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
            
        logger.info(f"All seeds set to {seed}")
        
    def hash_object(self, obj: Any) -> str:
        """Create deterministic hash of any object"""
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            # Convert to bytes for hashing
            if isinstance(obj, torch.Tensor):
                obj = obj.detach().cpu().numpy()
            # Round to avoid floating point precision issues
            obj = np.round(obj.astype(np.float32), decimals=6)
            return hashlib.sha256(obj.tobytes()).hexdigest()
        elif isinstance(obj, dict):
            # Sort keys for consistent ordering
            sorted_dict = json.dumps(obj, sort_keys=True, default=str)
            return hashlib.sha256(sorted_dict.encode()).hexdigest()
        elif isinstance(obj, (list, tuple)):
            # Hash each element
            combined = ''.join(self.hash_object(item) for item in obj)
            return hashlib.sha256(combined.encode()).hexdigest()
        else:
            # Convert to string and hash
            return hashlib.sha256(str(obj).encode()).hexdigest()
    
    def test_embedder_determinism(self, video_path: str) -> DeterminismTest:
        """Test V-JEPA/CLIP embedder determinism"""
        from src.embedders.vjepa_embedder import VJEPAEmbedder
        
        input_hash = self.hash_object(video_path)
        
        # Run 1
        self.set_all_seeds()
        embedder1 = VJEPAEmbedder()
        start1 = time.time()
        segments1, meta1 = embedder1.embed_segments(
            video_path, fps=1.0, window=16, strategy="temp_attn"
        )
        time1 = time.time() - start1
        
        # Extract embeddings for hashing
        embeddings1 = [seg["emb"] for seg in segments1]
        output_hash1 = self.hash_object(embeddings1)
        
        # Run 2 with same seed
        self.set_all_seeds()
        embedder2 = VJEPAEmbedder()
        start2 = time.time()
        segments2, meta2 = embedder2.embed_segments(
            video_path, fps=1.0, window=16, strategy="temp_attn"
        )
        time2 = time.time() - start2
        
        embeddings2 = [seg["emb"] for seg in segments2]
        output_hash2 = self.hash_object(embeddings2)
        
        is_deterministic = output_hash1 == output_hash2
        
        return DeterminismTest(
            component="vjepa_embedder",
            input_hash=input_hash,
            output_hash_run1=output_hash1,
            output_hash_run2=output_hash2,
            is_deterministic=is_deterministic,
            execution_time_run1=time1,
            execution_time_run2=time2,
            details={
                "num_segments": len(segments1),
                "embedding_dim": len(embeddings1[0]) if embeddings1 else 0,
                "fps": 1.0,
                "window": 16
            }
        )
    
    def test_director_determinism(self, video_path: str) -> DeterminismTest:
        """Test director module determinism"""
        from src.director.creative_director import analyze_video
        
        input_hash = self.hash_object(video_path)
        
        # Run 1
        self.set_all_seeds()
        start1 = time.time()
        results1 = analyze_video(video_path)
        time1 = time.time() - start1
        output_hash1 = self.hash_object(results1)
        
        # Run 2
        self.set_all_seeds()
        start2 = time.time()
        results2 = analyze_video(video_path)
        time2 = time.time() - start2
        output_hash2 = self.hash_object(results2)
        
        is_deterministic = output_hash1 == output_hash2
        
        return DeterminismTest(
            component="director",
            input_hash=input_hash,
            output_hash_run1=output_hash1,
            output_hash_run2=output_hash2,
            is_deterministic=is_deterministic,
            execution_time_run1=time1,
            execution_time_run2=time2,
            details={
                "modules_tested": list(results1.keys()) if results1 else []
            }
        )
    
    def test_transcription_determinism(self, audio_path: str) -> DeterminismTest:
        """Test transcription determinism"""
        from src.ops.transcribe import transcribe_audio
        
        input_hash = self.hash_object(audio_path)
        
        # Run 1
        self.set_all_seeds()
        start1 = time.time()
        transcript1 = transcribe_audio(audio_path)
        time1 = time.time() - start1
        
        # Extract text for comparison (timing might vary slightly)
        text1 = ' '.join(seg.get('text', '') for seg in transcript1.get('segments', []))
        output_hash1 = self.hash_object(text1)
        
        # Run 2
        self.set_all_seeds()
        start2 = time.time()
        transcript2 = transcribe_audio(audio_path)
        time2 = time.time() - start2
        
        text2 = ' '.join(seg.get('text', '') for seg in transcript2.get('segments', []))
        output_hash2 = self.hash_object(text2)
        
        # Text should be identical even if timing varies slightly
        is_deterministic = output_hash1 == output_hash2
        
        return DeterminismTest(
            component="transcription",
            input_hash=input_hash,
            output_hash_run1=output_hash1,
            output_hash_run2=output_hash2,
            is_deterministic=is_deterministic,
            execution_time_run1=time1,
            execution_time_run2=time2,
            details={
                "language": transcript1.get('language', 'unknown'),
                "num_segments": len(transcript1.get('segments', []))
            }
        )
    
    def test_silence_detection_determinism(self, video_path: str) -> DeterminismTest:
        """Test silence detection determinism"""
        from src.ops.silence import detect_silence
        
        input_hash = self.hash_object(video_path)
        
        # Run 1
        self.set_all_seeds()
        start1 = time.time()
        cuts1 = detect_silence(video_path)
        time1 = time.time() - start1
        output_hash1 = self.hash_object(cuts1)
        
        # Run 2
        self.set_all_seeds()
        start2 = time.time()
        cuts2 = detect_silence(video_path)
        time2 = time.time() - start2
        output_hash2 = self.hash_object(cuts2)
        
        is_deterministic = output_hash1 == output_hash2
        
        return DeterminismTest(
            component="silence_detection",
            input_hash=input_hash,
            output_hash_run1=output_hash1,
            output_hash_run2=output_hash2,
            is_deterministic=is_deterministic,
            execution_time_run1=time1,
            execution_time_run2=time2,
            details={
                "num_segments": len(cuts1) if isinstance(cuts1, list) else 0
            }
        )
    
    def test_broll_selection_determinism(self, video_path: str, query: str) -> DeterminismTest:
        """Test B-roll selection determinism"""
        from src.broll.selector import select_broll
        
        input_hash = self.hash_object((video_path, query))
        
        # Run 1
        self.set_all_seeds()
        start1 = time.time()
        selections1 = select_broll(video_path, query)
        time1 = time.time() - start1
        output_hash1 = self.hash_object(selections1)
        
        # Run 2
        self.set_all_seeds()
        start2 = time.time()
        selections2 = select_broll(video_path, query)
        time2 = time.time() - start2
        output_hash2 = self.hash_object(selections2)
        
        is_deterministic = output_hash1 == output_hash2
        
        return DeterminismTest(
            component="broll_selection",
            input_hash=input_hash,
            output_hash_run1=output_hash1,
            output_hash_run2=output_hash2,
            is_deterministic=is_deterministic,
            execution_time_run1=time1,
            execution_time_run2=time2,
            details={
                "query": query,
                "num_selections": len(selections1) if isinstance(selections1, list) else 0
            }
        )
    
    def test_shortsify_determinism(self, video_path: str) -> DeterminismTest:
        """Test shorts generation determinism"""
        from src.ops.shortsify import generate_shorts
        
        input_hash = self.hash_object(video_path)
        
        # Run 1
        self.set_all_seeds()
        start1 = time.time()
        shorts1 = generate_shorts(video_path)
        time1 = time.time() - start1
        
        # Hash the segments (not the actual video files)
        segments1 = shorts1.get('segments', []) if isinstance(shorts1, dict) else []
        output_hash1 = self.hash_object(segments1)
        
        # Run 2
        self.set_all_seeds()
        start2 = time.time()
        shorts2 = generate_shorts(video_path)
        time2 = time.time() - start2
        
        segments2 = shorts2.get('segments', []) if isinstance(shorts2, dict) else []
        output_hash2 = self.hash_object(segments2)
        
        is_deterministic = output_hash1 == output_hash2
        
        return DeterminismTest(
            component="shortsify",
            input_hash=input_hash,
            output_hash_run1=output_hash1,
            output_hash_run2=output_hash2,
            is_deterministic=is_deterministic,
            execution_time_run1=time1,
            execution_time_run2=time2,
            details={
                "num_shorts": len(segments1)
            }
        )
    
    def run_all_tests(self, test_video: str = "assets/pilots/test_video.mp4") -> Dict[str, Any]:
        """Run determinism tests on all components"""
        logger.info("="*60)
        logger.info("Running Determinism Validation Tests")
        logger.info("="*60)
        
        if not Path(test_video).exists():
            # Use any available video
            import glob
            videos = glob.glob("assets/pilots/*.mp4")
            if videos:
                test_video = videos[0]
                logger.info(f"Using test video: {test_video}")
        
        # Test each component
        tests_to_run = [
            ("Embedder", lambda: self.test_embedder_determinism(test_video)),
            ("Director", lambda: self.test_director_determinism(test_video)),
            ("Transcription", lambda: self.test_transcription_determinism(test_video)),
            ("Silence Detection", lambda: self.test_silence_detection_determinism(test_video)),
            ("B-roll Selection", lambda: self.test_broll_selection_determinism(test_video, "aerial view")),
            ("Shortsify", lambda: self.test_shortsify_determinism(test_video))
        ]
        
        for test_name, test_func in tests_to_run:
            logger.info(f"\nTesting {test_name}...")
            try:
                result = test_func()
                self.test_results.append(result)
                
                if result.is_deterministic:
                    logger.info(f"  ✅ {test_name} is DETERMINISTIC")
                else:
                    logger.error(f"  ❌ {test_name} is NON-DETERMINISTIC")
                    logger.error(f"     Hash1: {result.output_hash_run1[:16]}...")
                    logger.error(f"     Hash2: {result.output_hash_run2[:16]}...")
                
            except Exception as e:
                logger.error(f"  ⚠️ {test_name} test failed: {e}")
                self.test_results.append(
                    DeterminismTest(
                        component=test_name.lower().replace(" ", "_"),
                        input_hash="",
                        output_hash_run1="",
                        output_hash_run2="",
                        is_deterministic=False,
                        execution_time_run1=0,
                        execution_time_run2=0,
                        details={"error": str(e)}
                    )
                )
        
        # Generate summary
        summary = self.generate_summary()
        
        # Record in telemetry
        self.record_telemetry(summary)
        
        return summary
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary of determinism tests"""
        total_tests = len(self.test_results)
        deterministic_tests = sum(1 for t in self.test_results if t.is_deterministic)
        
        summary = {
            "total_tests": total_tests,
            "deterministic": deterministic_tests,
            "non_deterministic": total_tests - deterministic_tests,
            "all_deterministic": deterministic_tests == total_tests,
            "percentage": (deterministic_tests / total_tests * 100) if total_tests > 0 else 0,
            "details": []
        }
        
        for test in self.test_results:
            summary["details"].append({
                "component": test.component,
                "deterministic": test.is_deterministic,
                "time_variance": abs(test.execution_time_run2 - test.execution_time_run1),
                "details": test.details
            })
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("Determinism Validation Summary")
        logger.info("="*60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Deterministic: {deterministic_tests}")
        logger.info(f"Non-Deterministic: {total_tests - deterministic_tests}")
        logger.info(f"Success Rate: {summary['percentage']:.1f}%")
        
        if summary["all_deterministic"]:
            logger.info("\n✅ ALL COMPONENTS ARE DETERMINISTIC")
        else:
            logger.error("\n❌ SOME COMPONENTS ARE NON-DETERMINISTIC")
            logger.error("Non-deterministic components:")
            for test in self.test_results:
                if not test.is_deterministic:
                    logger.error(f"  - {test.component}")
        
        return summary
    
    def record_telemetry(self, summary: Dict[str, Any]):
        """Record determinism test results in telemetry"""
        try:
            from src.utils.telemetry import get_telemetry
            telemetry = get_telemetry()
            
            for test in self.test_results:
                telemetry.record_determinism_check(
                    input_hash=test.input_hash,
                    output_hash=test.output_hash_run1,  # Use first run hash
                    run_number=1,
                    outputs_identical=test.is_deterministic
                )
            
            # Record overall summary
            telemetry.record_metric(
                name="determinism_validation",
                category="validation",
                all_deterministic=summary["all_deterministic"],
                percentage=summary["percentage"],
                total_tests=summary["total_tests"]
            )
            
        except ImportError:
            logger.warning("Telemetry module not available")
    
    def save_report(self, output_path: str = "proof_pack/determinism_report.json"):
        """Save detailed determinism report"""
        report = {
            "timestamp": time.time(),
            "seed": self.seed,
            "summary": self.generate_summary(),
            "test_results": [
                {
                    "component": t.component,
                    "input_hash": t.input_hash,
                    "output_hash_run1": t.output_hash_run1,
                    "output_hash_run2": t.output_hash_run2,
                    "is_deterministic": t.is_deterministic,
                    "execution_time_run1": t.execution_time_run1,
                    "execution_time_run2": t.execution_time_run2,
                    "time_variance": abs(t.execution_time_run2 - t.execution_time_run1),
                    "details": t.details
                }
                for t in self.test_results
            ]
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nDeterminism report saved to {output_path}")
        return report


def main():
    """Run determinism validation"""
    validator = DeterminismValidator(seed=1234)
    summary = validator.run_all_tests()
    validator.save_report()
    
    # Exit with error code if not all deterministic
    if not summary["all_deterministic"]:
        logger.error("Determinism validation failed - some components are non-deterministic")
        exit(1)
    else:
        logger.info("Determinism validation passed - all components are deterministic")
        exit(0)


if __name__ == "__main__":
    main()