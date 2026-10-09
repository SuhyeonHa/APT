import torch
from .abstract_class import AbstractEvaluator


class PixelIOU(AbstractEvaluator):
    def __init__(self, threshold=0.5, mode="origin") -> None:
        self.name = "pixel-level IOU"
        self.desc = "pixel-level IOU"
        self.threshold = threshold
        self.mode = mode
    
    def Cal_IOU(self, predict, mask, shape_mask=None):
        if shape_mask is not None:
            predict = predict * shape_mask
            mask = mask * shape_mask
        
        predict = (predict > self.threshold).float().flatten(1)
        mask = mask.flatten(1)

        # Compute intersection and union
        intersection = torch.sum(predict * mask, dim=1)
        union = torch.sum(predict,dim=1) + torch.sum(mask,dim=1) - intersection

        iou = intersection / (union + 1e-8)  # Add small value to avoid division by zero

        return iou

    def Cal_IOU_2(self, predict, mask, shape_mask=None):
        predict = (predict > self.threshold).float().to(torch.int8)
        mask = mask.to(torch.int8)
        predict = 1 - predict
        mask = 1 - mask
        if shape_mask is not None:
            predict = predict * shape_mask.to(torch.int8)
            mask = mask * shape_mask.to(torch.int8)

        # Flatten the tensors
        predict = predict.flatten(1)
        mask = mask.flatten(1)
        print(predict.shape)
        # Compute intersection and union
        intersection = torch.sum(predict * mask, dim=1)
        union = torch.sum(predict,dim=1) + torch.sum(mask,dim=1) - intersection

        iou = intersection / (union + 1e-8)  # Add small value to avoid division by zero

        return iou
    
    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        if self.mode == "origin":
            IOU = self.Cal_IOU(predict, mask, shape_mask)
        elif self.mode == "reverse":
            IOU = self.Cal_IOU(1 - predict, mask, shape_mask)
        elif self.mode == "double":
            normal_iou = self.Cal_IOU(predict, mask, shape_mask)
            reverse_iou = self.Cal_IOU(1 - predict, mask, shape_mask)
            IOU = torch.max(normal_iou, reverse_iou)
        else:
            raise RuntimeError(f"Cal_AUC no mode name {self.mode}")
        return IOU
    def epoch_update(self):

        return None
    
    def recovery(self):
        return None
