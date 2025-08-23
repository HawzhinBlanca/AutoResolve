"""
AutoResolve v3.0 Compliance Monitoring System
Continuous validation of blueprint requirements
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceMonitor:
    """Monitor and enforce blueprint compliance requirements"""
    
    COMPLIANCE_RULES = {
        # Retrieval Performance Gates
        "vjepa_performance": {
            "metric": "sec_per_min",
            "threshold": 5.0,
            "operator": "<=",
            "alert": "V-JEPA performance degraded: {value:.2f} sec/min > 5.0 limit",
            "critical": True
        },
        "vjepa_quality_top3": {
            "metric": "top3_gain",
            "threshold": 0.15,
            "operator": ">=",
            "alert": "V-JEPA Top-3 gain {value:.2%} below 15% requirement",
            "critical": True
        },
        "vjepa_quality_mrr": {
            "metric": "mrr_gain",
            "threshold": 0.15,
            "operator": ">=",
            "alert": "V-JEPA MRR gain {value:.2%} below 15% requirement",
            "critical": True
        },
        
        # Memory Safety Gates
        "memory_usage": {
            "metric": "peak_rss_gb",
            "threshold": 16.0,
            "operator": "<=",
            "alert": "Memory usage {value:.1f}GB exceeded 16GB limit",
            "critical": True
        },
        "memory_adaptive_degrade": {
            "metric": "degradation_triggered",
            "threshold": False,
            "operator": "==",
            "alert": "Memory degradation was triggered - check optimization",
            "critical": False
        },
        
        # Director Quality Gates
        "director_f1": {
            "metric": "f1_iou0.5",
            "threshold": 0.60,
            "operator": ">=",
            "alert": "Director F1@IoU0.5 {value:.2f} below 0.60 requirement",
            "critical": False  # Can hide module if fails
        },
        "director_pr_auc": {
            "metric": "pr_auc",
            "threshold": 0.65,
            "operator": ">=",
            "alert": "Director PR-AUC {value:.2f} below 0.65 requirement",
            "critical": False
        },
        "director_performance": {
            "metric": "director_sec_per_min",
            "threshold": 7.5,
            "operator": "<=",
            "alert": "Director analysis {value:.1f} sec/min > 7.5 limit",
            "critical": False
        },
        
        # Operations Gates
        "transcription_speed": {
            "metric": "rtf",
            "threshold": 1.5,
            "operator": "<=",
            "alert": "Transcription RTF {value:.2f} slower than 1.5x realtime",
            "critical": True
        },
        "silence_false_cut": {
            "metric": "false_cut_rate",
            "threshold": 0.05,
            "operator": "<=",
            "alert": "Silence cutter false-cut rate {value:.2%} > 5% limit",
            "critical": True
        },
        "shortsify_latency": {
            "metric": "shorts_sec_30min",
            "threshold": 120,
            "operator": "<=",
            "alert": "Shortsify took {value}s > 120s limit for 30-min video",
            "critical": True
        },
        
        # B-roll Quality Gates
        "broll_top3_match": {
            "metric": "broll_top3",
            "threshold": 0.65,
            "operator": ">=",
            "alert": "B-roll Top-3 match rate {value:.2%} below 65%",
            "critical": True
        },
        "broll_placement_conflicts": {
            "metric": "placement_conflict_rate",
            "threshold": 0.10,
            "operator": "<=",
            "alert": "B-roll placement conflicts {value:.2%} > 10% limit",
            "critical": True
        },
        
        # Determinism Requirements
        "determinism_check": {
            "metric": "outputs_identical",
            "threshold": True,
            "operator": "==",
            "alert": "Non-deterministic outputs detected - seeds not working",
            "critical": True
        }
    }
    
    def __init__(self, metrics_path: str = "artifacts/metrics.jsonl"):
        self.metrics_path = Path(metrics_path)
        self.violations: List[Dict[str, Any]] = []
        self.last_check = time.time()
        
    def check_metric(self, rule_name: str, value: Any) -> Tuple[bool, str]:
        """Check if a metric passes its compliance rule"""
        if rule_name not in self.COMPLIANCE_RULES:
            return True, ""
            
        rule = self.COMPLIANCE_RULES[rule_name]
        threshold = rule["threshold"]
        operator = rule["operator"]
        
        # Perform comparison based on operator
        passed = False
        if operator == "<=":
            passed = value <= threshold
        elif operator == ">=":
            passed = value >= threshold
        elif operator == "==":
            passed = value == threshold
        elif operator == "!=":
            passed = value != threshold
            
        if not passed:
            alert = rule["alert"].format(value=value)
            return False, alert
            
        return True, ""
    
    def load_latest_metrics(self) -> Dict[str, Any]:
        """Load the most recent metrics from metrics.jsonl"""
        if not self.metrics_path.exists():
            return {}
            
        metrics = {}
        with open(self.metrics_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    # Flatten nested metrics
                    if "name" in entry and "metrics" in entry:
                        for key, value in entry["metrics"].items():
                            metrics[f"{entry['name']}_{key}"] = value
                    else:
                        metrics.update(entry)
                except json.JSONDecodeError:
                    continue
                    
        return metrics
    
    def run_compliance_check(self) -> Dict[str, Any]:
        """Run full compliance check against all rules"""
        metrics = self.load_latest_metrics()
        results = {
            "timestamp": time.time(),
            "passed": True,
            "critical_violations": [],
            "warnings": [],
            "metrics_checked": 0,
            "compliance_score": 0.0
        }
        
        total_rules = len(self.COMPLIANCE_RULES)
        passed_rules = 0
        
        for rule_name, rule in self.COMPLIANCE_RULES.items():
            metric_name = rule["metric"]
            
            # Try to find the metric in loaded data
            value = None
            for key in metrics:
                if metric_name in key:
                    value = metrics[key]
                    break
                    
            if value is None:
                # Metric not found - this is a violation
                alert = f"Required metric '{metric_name}' not found in telemetry"
                if rule["critical"]:
                    results["critical_violations"].append({
                        "rule": rule_name,
                        "alert": alert
                    })
                    results["passed"] = False
                else:
                    results["warnings"].append({
                        "rule": rule_name,
                        "alert": alert
                    })
            else:
                # Check the metric against its rule
                passed, alert = self.check_metric(rule_name, value)
                results["metrics_checked"] += 1
                
                if passed:
                    passed_rules += 1
                else:
                    violation = {
                        "rule": rule_name,
                        "metric": metric_name,
                        "value": value,
                        "threshold": rule["threshold"],
                        "alert": alert
                    }
                    
                    if rule["critical"]:
                        results["critical_violations"].append(violation)
                        results["passed"] = False
                    else:
                        results["warnings"].append(violation)
                        
        # Calculate compliance score
        results["compliance_score"] = (passed_rules / total_rules) * 100
        
        # Log results
        self._log_results(results)
        
        return results
    
    def _log_results(self, results: Dict[str, Any]):
        """Log compliance check results"""
        if results["passed"]:
            logger.info(f"✅ Compliance check PASSED - Score: {results['compliance_score']:.1f}%")
        else:
            logger.error(f"❌ Compliance check FAILED - Score: {results['compliance_score']:.1f}%")
            
        if results["critical_violations"]:
            logger.error(f"Critical violations: {len(results['critical_violations'])}")
            for v in results["critical_violations"]:
                logger.error(f"  - {v['alert']}")
                
        if results["warnings"]:
            logger.warning(f"Warnings: {len(results['warnings'])}")
            for w in results["warnings"]:
                logger.warning(f"  - {w['alert']}")
    
    def generate_compliance_report(self, output_path: str = "proof_pack/compliance_report.json"):
        """Generate detailed compliance report"""
        results = self.run_compliance_check()
        
        # Add detailed breakdown
        results["rule_details"] = {}
        for rule_name, rule in self.COMPLIANCE_RULES.items():
            results["rule_details"][rule_name] = {
                "description": rule["alert"].split("{")[0].strip(),
                "threshold": rule["threshold"],
                "operator": rule["operator"],
                "critical": rule["critical"]
            }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Compliance report saved to {output_path}")
        return results
    
    def enforce_regression_protection(self, metrics: Dict[str, Any]) -> bool:
        """
        Enforce that no metrics regress from baseline.
        Returns False if regression detected.
        """
        baseline_path = Path("proof_pack/baseline_metrics.json")
        
        if not baseline_path.exists():
            # No baseline yet - save current as baseline
            with open(baseline_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info("Baseline metrics established")
            return True
            
        # Load baseline
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
            
        # Check for regressions
        regressions = []
        for key, current_value in metrics.items():
            if key in baseline:
                baseline_value = baseline[key]
                
                # Determine if this metric should increase or decrease
                if any(x in key for x in ["error", "loss", "latency", "memory", "sec_per_min"]):
                    # Lower is better
                    if current_value > baseline_value * 1.05:  # 5% tolerance
                        regressions.append({
                            "metric": key,
                            "baseline": baseline_value,
                            "current": current_value,
                            "regression": f"+{((current_value/baseline_value - 1) * 100):.1f}%"
                        })
                elif any(x in key for x in ["accuracy", "f1", "auc", "top3", "mrr"]):
                    # Higher is better
                    if current_value < baseline_value * 0.95:  # 5% tolerance
                        regressions.append({
                            "metric": key,
                            "baseline": baseline_value,
                            "current": current_value,
                            "regression": f"{((current_value/baseline_value - 1) * 100):.1f}%"
                        })
                        
        if regressions:
            logger.error(f"❌ Regression detected in {len(regressions)} metrics:")
            for r in regressions:
                logger.error(f"  - {r['metric']}: {r['baseline']:.3f} → {r['current']:.3f} ({r['regression']})")
            return False
            
        logger.info("✅ No regressions detected")
        return True


def main():
    """Run compliance monitoring"""
    monitor = ComplianceMonitor()
    
    # Run compliance check
    results = monitor.run_compliance_check()
    
    # Generate report
    monitor.generate_compliance_report()
    
    # Exit with error code if compliance failed
    if not results["passed"]:
        logger.error("Compliance check failed - project not ready for production")
        exit(1)
    else:
        logger.info(f"Compliance check passed with score: {results['compliance_score']:.1f}%")
        exit(0)


if __name__ == "__main__":
    main()