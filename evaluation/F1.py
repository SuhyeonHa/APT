import torch
from .abstract_class import AbstractEvaluator


class PixelF1(AbstractEvaluator):
    def __init__(self, threshold = 0.5, mode = "origin") -> None:
        super().__init__()
        self.name = "pixel-level F1"
        self.desc = "pixel-level F1"
        self.threshold = threshold
        self.image_num = 0
        #  mode : "origin, reverse, double"
        self.mode = mode

    def Cal_Confusion_Matrix(self, predict, mask, shape_mask):
        """compute local confusion matrix for a batch of predict and target masks
        Args:
            predict (_type_): _description_
            mask (_type_): _description_
            region (_type_): _description_
            
        Returns:
            TP, TN, FP, FN
        """
        threshold = self.threshold
        predict = (predict > threshold).float()
        if(shape_mask != None):
            TP = torch.sum(predict * mask * shape_mask, dim=(1, 2, 3))
            TN = torch.sum((1-predict) * (1-mask) * shape_mask, dim=(1, 2, 3))
            FP = torch.sum(predict * (1-mask) * shape_mask, dim=(1, 2, 3))
            FN = torch.sum((1-predict) * mask * shape_mask, dim=(1, 2, 3))
        else:
            TP = torch.sum(predict * mask, dim=(1, 2, 3))  
            TN = torch.sum((1-predict) * (1-mask), dim=(1, 2, 3)) 
            FP = torch.sum(predict * (1-mask), dim=(1, 2, 3)) 
            FN = torch.sum((1-predict) * mask, dim=(1, 2, 3))         
        return TP, TN, FP, FN

    def Cal_Reverse_Confusion_Matrix(self, predict, mask, shape_mask):
        """compute local confusion matrix for a batch of predict and target masks
        Args:
            predict (_type_): _description_
            mask (_type_): _description_
            region (_type_): _description_
            
        Returns:
            TP, TN, FP, FN
        """
        threshold = self.threshold
        predict = (predict > threshold).float()
        if(shape_mask != None):
            TP = torch.sum((1-predict) * mask * shape_mask, dim=(1, 2, 3))
            TN = torch.sum(predict * (1-mask) * shape_mask, dim=(1, 2, 3))
            FP = torch.sum((1-predict) * (1-mask) * shape_mask, dim=(1, 2, 3))
            FN = torch.sum(predict * mask * shape_mask, dim=(1, 2, 3))
        else:
            TP = torch.sum((1-predict) * mask, dim=(1, 2, 3))
            TN = torch.sum(predict * (1-mask), dim=(1, 2, 3))
            FP = torch.sum((1-predict) * (1-mask), dim=(1, 2, 3))
            FN = torch.sum(predict * mask, dim=(1, 2, 3))
        return TP, TN, FP, FN

    def Cal_F1(self, TP, TN, FP, FN):
        """_summary_

        Args:
            TP (_type_): _description_
            TN (_type_): _description_
            FP (_type_): _description_
            FN (_type_): _description_

        Returns:
            _type_: _description_
        """
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        # F1 = torch.mean(F1) # fuse the Batch dimension
        return F1

    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        if self.mode == "origin":
            TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
            F1 = self.Cal_F1(TP, TN, FP, FN)
        elif self.mode == "reverse":
            TP, TN, FP, FN = self.Cal_Reverse_Confusion_Matrix(predict, mask, shape_mask)
            F1 = self.Cal_F1(TP, TN, FP, FN)
        elif self.mode == "double":
            # todo
            TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
            F1 = torch.max(self.Cal_F1(TP, TN, FP, FN), self.Cal_F1(FN, FP, TN, TP))
        else:
            raise RuntimeError(f"Cal_F1 no mode name {self.mode}")
        
        return F1
    def epoch_update(self):

        return None
    def recovery(self):
        self.image_num = 0

    def Cal_FPR(self, predict, mask, shape_mask=None):
        TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
        return FP / (FP + TN)

