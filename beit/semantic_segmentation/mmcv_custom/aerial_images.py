from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset



@DATASETS.register_module()
class AerialImages(CustomDataset):
    CLASSES = (
        "background", "building"
    )
    PALETTE = [[56, 56, 56], [122, 122, 122]]
    def __init__(self, classes, palette, **kwargs):
        super().__init__(img_suffix='.tiff', seg_map_suffix='.tiff', ignore_index=255, **kwargs)
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(classes, palette)
