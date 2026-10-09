import torch
from .abstract_class import AbstractEvaluator


class PixelAUC(AbstractEvaluator):
    def __init__(self, threshold=0.5, mode="origin") -> None:
        self.name = "pixel-level AUC"
        self.desc = "pixel-level AUC"
        self.threshold = threshold
        self.mode = mode

    def Cal_AUC(self, y_true, y_scores, shape_mask=None):
        if shape_mask is not None:
            y_true = y_true * shape_mask
            y_scores = y_scores * shape_mask
        
        y_true = y_true.flatten()
        y_scores = y_scores.flatten()

        if shape_mask is not None:
            valid_mask = shape_mask.flatten() > 0
            y_true = y_true[valid_mask]
            y_scores = y_scores[valid_mask]

        desc_score_indices = torch.argsort(y_scores, descending=True)
        y_true_sorted = y_true[desc_score_indices]

        n_pos = torch.sum(y_true_sorted).item()
        n_neg = len(y_true_sorted) - n_pos

        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        tpr = tps / n_pos
        fpr = fps / n_neg

        auc = torch.trapz(tpr, fpr)

        return auc.item()
        
    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        # TODO
        AUC_list = []
        if self.mode == "origin":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask == None else shape_mask[idx]
                AUC_list.append(self.Cal_AUC(mask[idx], predict[idx], single_shape_mask))
        elif self.mode == "reverse":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask == None else shape_mask[idx]
                AUC_list.append(self.Cal_AUC(mask[idx], 1 - predict[idx], single_shape_mask))
        elif self.mode == "double":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask == None else shape_mask[idx]
                AUC_list.append(max(self.Cal_AUC(mask[idx], predict[idx], single_shape_mask), self.Cal_AUC(mask[idx], 1 - predict[idx], single_shape_mask)))
        else:
            raise RuntimeError(f"Cal_AUC no mode name {self.mode}")
        
        return torch.tensor(AUC_list)

    def epoch_update(self):

        return None
    
    def recovery(self):
        return None

