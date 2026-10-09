"""
Base interface for pixel-level evaluators. Image-level and pixel-level
metrics differ in how they are computed, hence a shared abstract class.
"""
class AbstractEvaluator(object):
    def __init__(self) -> None:
        self.name = None
        self.desc = None
        self.threshold = None
    def batch_update(self, predict, pred_label, mask, shape_mask=None, *args, **kwargs):
        """Update running statistics at the end of each batch."""
        raise NotImplementedError
    def epoch_update(self):
        """Aggregate statistics collected over the epoch and return the final metric."""
        raise NotImplementedError
    def recovery(self):
        raise NotImplementedError




    
