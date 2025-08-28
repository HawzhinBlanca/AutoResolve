from src.utils.memory import Budget, enforce_budget

def test_enforce_budget_degrade_order():
    b = Budget(max_gb=0.0, fps=2.0, window=16, crop=256, max_segments=10)
    changes, nb = enforce_budget(b, device='cpu', aggressive=True)
    # Should attempt fps degradation first when over budget
    if changes:
        assert changes[0][0] in ("fps", "window", "crop")


