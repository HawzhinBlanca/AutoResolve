import logging
import json
import time
import os
import sys
from src.utils.common import set_global_seed, iou

logger = logging.getLogger(__name__)

"""
Blueprint3 B-roll Module - Placer
Real implementation meeting <10% placement conflicts requirement
"""

class BrollPlacer:
    def __init__(self):
        set_global_seed(1234)
        self.max_conflict_rate = 0.10  # Blueprint requirement: <10%
        
    def place_broll(self, selection_data, transcript_data=None, output_path=None):
        """
        Place B-roll clips with conflict avoidance
        Returns: (placement_data, performance_metrics)
        """
        start_time = time.time()
        
        if "selections" not in selection_data:
            return {"error": "No selections in input data"}, {"success": False}
        
        # Extract dialogue/emphasis regions to avoid
        protected_regions = self._extract_protected_regions(transcript_data)
        
        # Generate placement timeline
        placements = []
        conflicts = 0
        total_placements = 0
        
        for selection in selection_data["selections"]:
            main_segment = selection["main_segment_time"]
            selected_clip = selection["selected_clip"]
            
            # Determine optimal placement within segment
            placement = self._find_optimal_placement(
                main_segment, 
                selected_clip,
                protected_regions,
                placements  # Existing placements to avoid overlap
            )
            
            if placement:
                # Check for conflicts
                has_conflict = self._check_conflicts(placement, protected_regions)
                if has_conflict:
                    conflicts += 1
                
                placements.append({
                    "id": f"broll_{len(placements)+1}",
                    "main_segment": main_segment,
                    "broll_clip": selected_clip,
                    "placement": {
                        "start_time": placement[0],
                        "end_time": placement[1],
                        "duration": placement[1] - placement[0]
                    },
                    "track": "V2",  # B-roll track
                    "opacity": 1.0,
                    "has_conflict": has_conflict,
                    "query_used": selection["query"]
                })
                total_placements += 1
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        conflict_rate = conflicts / total_placements if total_placements > 0 else 0.0
        
        metrics = {
            "processing_time_s": elapsed,
            "total_placements": total_placements,
            "conflicts": conflicts,
            "conflict_rate": conflict_rate,
            "meets_requirement": conflict_rate < self.max_conflict_rate,
            "protected_regions": len(protected_regions)
        }
        
        # Create placement data
        placement_data = {
            "version": "3.0",
            "source_selection": selection_data.get("main_video", "unknown"),
            "placements": placements,
            "protected_regions": protected_regions,
            "metadata": {
                "generated_at": time.time(),
                "conflict_rate": conflict_rate,
                "total_placements": total_placements,
                "algorithm": "conflict_avoidance_v1"
            }
        }
        
        # Save placement data
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(placement_data, f, indent=2)
        
        return placement_data, metrics
    
    def _extract_protected_regions(self, transcript_data):
        """Extract dialogue and emphasis regions that should not be covered"""
        protected_regions = []
        
        if not transcript_data or "segments" not in transcript_data:
            return protected_regions
        
        for segment in transcript_data["segments"]:
            # Protect dialogue segments
            text = segment["text"].strip()
            if len(text) > 5:  # Meaningful dialogue
                # Add some padding around dialogue
                padding = 0.2  # 200ms padding
                start = max(0, segment["start"] - padding)
                end = segment["end"] + padding
                
                protected_regions.append({
                    "start": start,
                    "end": end,
                    "type": "dialogue",
                    "text": text[:50] + "..." if len(text) > 50 else text
                })
        
        # Merge overlapping regions
        protected_regions = self._merge_overlapping_regions(protected_regions)
        
        return protected_regions
    
    def _merge_overlapping_regions(self, regions):
        """Merge overlapping protected regions"""
        if not regions:
            return []
        
        # Sort by start time
        regions.sort(key=lambda x: x["start"])
        
        merged = [regions[0]]
        
        for current in regions[1:]:
            last = merged[-1]
            
            # Check for overlap
            if current["start"] <= last["end"]:
                # Merge regions
                last["end"] = max(last["end"], current["end"])
                last["type"] = "merged_dialogue"
                if "text" in current:
                    last["text"] = last.get("text", "") + " | " + current["text"]
            else:
                merged.append(current)
        
        return merged
    
    def _find_optimal_placement(self, main_segment, selected_clip, protected_regions, existing_placements):
        """Find optimal placement for B-roll clip within main segment"""
        segment_start, segment_end = main_segment
        segment_duration = segment_end - segment_start
        clip_duration = min(selected_clip["duration"], segment_duration * 0.8)  # Max 80% of segment
        
        # Try different positions within segment
        num_positions = 10
        best_position = None
        min_conflicts = float('inf')
        
        for i in range(num_positions):
            # Position within segment
            position_ratio = i / (num_positions - 1) if num_positions > 1 else 0.5
            latest_start = segment_end - clip_duration
            placement_start = segment_start + position_ratio * (latest_start - segment_start)
            placement_end = placement_start + clip_duration
            
            # Ensure within bounds
            if placement_end > segment_end:
                placement_start = segment_end - clip_duration
                placement_end = segment_end
            
            if placement_start < segment_start:
                continue
            
            # Count conflicts
            conflicts = 0
            
            # Check against protected regions
            for region in protected_regions:
                overlap = iou([placement_start, placement_end], [region["start"], region["end"]])
                if overlap > 0.1:  # 10% overlap threshold
                    conflicts += 1
            
            # Check against existing placements
            for existing in existing_placements:
                existing_time = [existing["placement"]["start_time"], existing["placement"]["end_time"]]
                overlap = iou([placement_start, placement_end], existing_time)
                if overlap > 0.0:  # No overlap with existing B-roll
                    conflicts += 10  # Heavy penalty
            
            # Update best position
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_position = (placement_start, placement_end)
        
        return best_position
    
    def _check_conflicts(self, placement, protected_regions):
        """Check if placement conflicts with protected regions"""
        placement_start, placement_end = placement
        
        for region in protected_regions:
            overlap = iou([placement_start, placement_end], [region["start"], region["end"]])
            if overlap > 0.1:  # 10% overlap is considered conflict
                return True
        
        return False

def placer_cli():
    """CLI interface for B-roll placement"""
    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.broll.placer <selection_file> [transcript_file] [output_path]")
        sys.exit(1)
    
    selection_file = sys.argv[1]
    transcript_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_path = sys.argv[3] if len(sys.argv) > 3 else "artifacts/broll/overlay.json"
    
    # Load selection data
    if not os.path.exists(selection_file):
        logger.info(f"Selection file not found: {selection_file}")
        sys.exit(1)
    
    with open(selection_file) as f:
        selection_data = json.load(f)
    
    # Load transcript if provided
    transcript_data = None
    if transcript_file and os.path.exists(transcript_file):
        with open(transcript_file) as f:
            transcript_data = json.load(f)
    
    # Place B-roll
    placer = BrollPlacer()
    placement_data, metrics = placer.place_broll(selection_data, transcript_data, output_path)
    
    if "error" in placement_data:
        logger.error(f"B-roll placement failed: {placement_data['error']}")
    else:
        logger.info("B-roll placement complete:")
        logger.info(f"  Processing time: {metrics['processing_time_s']:.1f}s")
        logger.info(f"  Total placements: {metrics['total_placements']}")
        logger.info(f"  Conflicts: {metrics['conflicts']}")
        logger.info(f"  Conflict rate: {metrics['conflict_rate']:.3f}")
        logger.info(f"  Meets requirement (<{placer.max_conflict_rate}): {metrics['meets_requirement']}")
        logger.info(f"  Protected regions: {metrics['protected_regions']}")
        logger.info(f"  Output: {output_path}")

if __name__ == "__main__":
    placer_cli()
