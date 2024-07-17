from .ho_moondream import Moondream
from .ho_moondream import BboxToMask

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique

NODE_CLASS_MAPPINGS = {
    "Moondream Interrogator": Moondream,
    "BboxToMask": BboxToMask,
}
