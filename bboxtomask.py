import torch
import numpy as np

class BboxToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox": ("BBOX",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "bbox_to_mask"
    CATEGORY = "Hangover"

    def bbox_to_mask(self, image, bbox):
        if bbox is None or len(bbox) != 4:
            raise ValueError("Invalid bbox format. Expected (x, y, width, height)")

        x, y, width, height = bbox
        
        # Get image dimensions
        c, h, w = image.shape
        
        # Create an empty mask
        mask = torch.zeros((h, w), dtype=torch.float32)
        
        # Set the bbox region to 1
        mask[y:y+height, x:x+width] = 1.0
        
        return (mask.unsqueeze(0),)  # Return as a 1,H,W tensor
