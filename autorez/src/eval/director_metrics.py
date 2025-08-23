"""
Real director quality metrics calculation
Implements F1@IoU0.5 and PR-AUC for narrative beat detection
"""

import numpy as np
import json
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BeatAnnotation:
    """Ground truth beat annotation"""
    t0: float
    t1: float
    beat_type: str  # incident, climax, resolution
    confidence: float = 1.0

@dataclass
class BeatPrediction:
    """Predicted beat from director module"""
    t0: float
    t1: float
    beat_type: str
    score: float  # confidence score

def calculate_iou(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """Calculate Intersection over Union between two time segments"""
    # Calculate intersection
    inter_start = max(pred[0], gt[0])
    inter_end = min(pred[1], gt[1])
    
    if inter_end <= inter_start:
        return 0.0
    
    intersection = inter_end - inter_start
    
    # Calculate union
    pred_duration = pred[1] - pred[0]
    gt_duration = gt[1] - gt[0]
    union = pred_duration + gt_duration - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union

def match_predictions_to_ground_truth(
    predictions: List[BeatPrediction],
    ground_truth: List[BeatAnnotation],
    iou_threshold: float = 0.5
) -> Tuple[List[bool], List[bool], List[float]]:
    """
    Match predictions to ground truth using IoU threshold.
    Returns: (true_positives, false_positives, scores)
    """
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)
    
    matched_gt = set()
    true_positives = []
    scores = []
    
    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            
            # Check if beat types match
            if pred.beat_type != gt.beat_type:
                continue
            
            iou = calculate_iou((pred.t0, pred.t1), (gt.t0, gt.t1))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if this is a true positive
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            true_positives.append(True)
            matched_gt.add(best_gt_idx)
        else:
            true_positives.append(False)
        
        scores.append(pred.score)
    
    # Calculate false negatives (unmatched ground truth)
    false_negatives = [i not in matched_gt for i in range(len(ground_truth))]
    
    return true_positives, false_negatives, scores

def calculate_f1_at_iou(
    predictions: List[BeatPrediction],
    ground_truth: List[BeatAnnotation],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate F1 score at specific IoU threshold.
    """
    if not predictions or not ground_truth:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    true_positives, false_negatives, _ = match_predictions_to_ground_truth(
        predictions, ground_truth, iou_threshold
    )
    
    tp_count = sum(true_positives)
    fp_count = len(true_positives) - tp_count
    fn_count = sum(false_negatives)
    
    # Calculate precision and recall
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    
    # Calculate F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count
    }

def calculate_pr_auc(
    predictions: List[BeatPrediction],
    ground_truth: List[BeatAnnotation],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculate Precision-Recall curve and Area Under Curve.
    """
    if not predictions or not ground_truth:
        return {"pr_auc": 0.0, "precision": [], "recall": []}
    
    # Get matches and scores
    true_positives, _, scores = match_predictions_to_ground_truth(
        predictions, ground_truth, iou_threshold
    )
    
    # Convert to binary labels (1 for TP, 0 for FP)
    y_true = np.array(true_positives, dtype=int)
    y_scores = np.array(scores)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate AUC
    pr_auc = auc(recall, precision)
    
    return {
        "pr_auc": pr_auc,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist()
    }

class DirectorEvaluator:
    """
    Evaluate director module predictions against ground truth.
    """
    
    def __init__(self, annotations_dir: str = "datasets/annotations"):
        self.annotations_dir = Path(annotations_dir)
        self.ground_truth = self._load_annotations()
        
    def _load_annotations(self) -> Dict[str, List[BeatAnnotation]]:
        """Load ground truth annotations from JSONL files"""
        ground_truth = {
            "incidents": [],
            "climax": [],
            "resolution": []
        }
        
        for beat_type in ground_truth.keys():
            annotation_file = self.annotations_dir / f"{beat_type}.jsonl"
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            ground_truth[beat_type].append(
                                BeatAnnotation(
                                    t0=data["t0"],
                                    t1=data["t1"],
                                    beat_type=beat_type,
                                    confidence=data.get("confidence", 1.0)
                                )
                            )
                        except (json.JSONDecodeError, KeyError):
                            continue
        
        return ground_truth
    
    def evaluate_predictions(
        self,
        predictions: Dict[str, List[Tuple[float, float]]],
        confidence_scores: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate director predictions against ground truth.
        
        Args:
            predictions: Dict mapping beat_type to list of (t0, t1) tuples
            confidence_scores: Optional dict of confidence scores for each prediction
        
        Returns:
            Evaluation metrics including F1@IoU0.5 and PR-AUC
        """
        results = {
            "overall": {"f1": 0.0, "pr_auc": 0.0},
            "by_type": {}
        }
        
        all_predictions = []
        all_ground_truth = []
        
        # Process each beat type
        for beat_type in ["incidents", "climax", "resolution"]:
            type_predictions = []
            type_ground_truth = self.ground_truth.get(beat_type, [])
            
            # Convert predictions to BeatPrediction objects
            if beat_type in predictions:
                for idx, (t0, t1) in enumerate(predictions[beat_type]):
                    score = 1.0
                    if confidence_scores and beat_type in confidence_scores:
                        if idx < len(confidence_scores[beat_type]):
                            score = confidence_scores[beat_type][idx]
                    
                    type_predictions.append(
                        BeatPrediction(t0=t0, t1=t1, beat_type=beat_type, score=score)
                    )
            
            # Calculate metrics for this beat type
            if type_predictions and type_ground_truth:
                f1_metrics = calculate_f1_at_iou(type_predictions, type_ground_truth)
                pr_metrics = calculate_pr_auc(type_predictions, type_ground_truth)
                
                results["by_type"][beat_type] = {
                    "f1": f1_metrics["f1"],
                    "precision": f1_metrics["precision"],
                    "recall": f1_metrics["recall"],
                    "pr_auc": pr_metrics["pr_auc"],
                    "predictions": len(type_predictions),
                    "ground_truth": len(type_ground_truth)
                }
            else:
                results["by_type"][beat_type] = {
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "pr_auc": 0.0,
                    "predictions": len(type_predictions),
                    "ground_truth": len(type_ground_truth)
                }
            
            # Aggregate for overall metrics
            all_predictions.extend(type_predictions)
            all_ground_truth.extend(type_ground_truth)
        
        # Calculate overall metrics
        if all_predictions and all_ground_truth:
            overall_f1 = calculate_f1_at_iou(all_predictions, all_ground_truth)
            overall_pr = calculate_pr_auc(all_predictions, all_ground_truth)
            
            results["overall"] = {
                "f1": overall_f1["f1"],
                "precision": overall_f1["precision"],
                "recall": overall_f1["recall"],
                "pr_auc": overall_pr["pr_auc"],
                "total_predictions": len(all_predictions),
                "total_ground_truth": len(all_ground_truth),
                "passes_f1_gate": overall_f1["f1"] >= 0.60,
                "passes_pr_auc_gate": overall_pr["pr_auc"] >= 0.65
            }
        
        return results
    
    def evaluate_from_file(self, predictions_file: str) -> Dict[str, Any]:
        """
        Evaluate predictions from a JSON file.
        
        Expected format:
        {
            "incidents": [[t0, t1], ...],
            "climax": [[t0, t1], ...],
            "resolution": [[t0, t1], ...]
        }
        """
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        
        # Extract predictions
        predictions = {}
        confidence_scores = {}
        
        for beat_type in ["incidents", "climax", "resolution"]:
            if beat_type in predictions_data:
                if isinstance(predictions_data[beat_type], list):
                    predictions[beat_type] = []
                    scores = []
                    
                    for item in predictions_data[beat_type]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            predictions[beat_type].append((item[0], item[1]))
                            # Use third element as score if available
                            scores.append(item[2] if len(item) > 2 else 1.0)
                        elif isinstance(item, dict):
                            predictions[beat_type].append((item["t0"], item["t1"]))
                            scores.append(item.get("score", 1.0))
                    
                    if scores:
                        confidence_scores[beat_type] = scores
        
        return self.evaluate_predictions(predictions, confidence_scores)
    
    def report_metrics(self, results: Dict[str, Any], output_file: Optional[str] = None):
        """Generate and optionally save evaluation report"""
        report = {
            "summary": {
                "f1_iou0.5": results["overall"]["f1"],
                "pr_auc": results["overall"]["pr_auc"],
                "precision": results["overall"]["precision"],
                "recall": results["overall"]["recall"],
                "passes_gates": (
                    results["overall"].get("passes_f1_gate", False) and
                    results["overall"].get("passes_pr_auc_gate", False)
                )
            },
            "detailed_results": results,
            "gate_requirements": {
                "f1_iou0.5_threshold": 0.60,
                "pr_auc_threshold": 0.65
            }
        }
        
        # Log results
        logger.info("="*50)
        logger.info("Director Evaluation Results")
        logger.info("="*50)
        logger.info(f"Overall F1@IoU0.5: {results['overall']['f1']:.3f} (gate: ≥0.60)")
        logger.info(f"Overall PR-AUC: {results['overall']['pr_auc']:.3f} (gate: ≥0.65)")
        logger.info(f"Overall Precision: {results['overall']['precision']:.3f}")
        logger.info(f"Overall Recall: {results['overall']['recall']:.3f}")
        logger.info("-"*50)
        
        for beat_type, metrics in results["by_type"].items():
            logger.info(f"{beat_type.capitalize()}:")
            logger.info(f"  F1: {metrics['f1']:.3f}, P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}")
            logger.info(f"  PR-AUC: {metrics['pr_auc']:.3f}")
            logger.info(f"  Predictions: {metrics['predictions']}, Ground Truth: {metrics['ground_truth']}")
        
        gate_status = "✅ PASSED" if report["summary"]["passes_gates"] else "❌ FAILED"
        logger.info(f"\nGate Status: {gate_status}")
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_file}")
        
        return report


def main():
    """Test the evaluator with sample data"""
    evaluator = DirectorEvaluator()
    
    # Create sample predictions for testing
    sample_predictions = {
        "incidents": [(30.0, 45.0), (120.0, 135.0), (200.0, 215.0)],
        "climax": [(180.0, 195.0)],
        "resolution": [(220.0, 240.0)]
    }
    
    # Evaluate
    results = evaluator.evaluate_predictions(sample_predictions)
    evaluator.report_metrics(results, "proof_pack/director_evaluation.json")
    
    # Record in telemetry
    from src.utils.telemetry import get_telemetry
    telemetry = get_telemetry()
    telemetry.record_director_evaluation(
        f1_iou05=results["overall"]["f1"],
        pr_auc=results["overall"]["pr_auc"],
        precision=results["overall"]["precision"],
        recall=results["overall"]["recall"],
        sec_per_min=3.5,  # Example timing
        module="narrative"
    )
    
    return results


if __name__ == "__main__":
    main()