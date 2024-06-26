from collections import OrderedDict

from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .xml_style import XMLDataset

@DATASETS.register_module()
class RDDDataset(XMLDataset):
    CLASSES = ('crater',)
    def __init__(self, **kwargs):
        super(RDDDataset, self).__init__(**kwargs)    