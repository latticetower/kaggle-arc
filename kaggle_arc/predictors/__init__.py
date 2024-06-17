import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from predictors.basic import IdPredictor, ZerosPredictor, ConstPredictor, FillPredictor
from predictors.complex import ComplexPredictor
from predictors.color_counting import ColorCountingPredictor
from predictors.shapes import (
    RepeatingPredictor,
    FractalPredictor,
    ResizingPredictor,
    MirrorPredictor,
    ConstantShaper,
)
from predictors.boosting_tree import (
    BoostingTreePredictor,
    BoostingTreePredictor2,
    BoostingTreePredictor3,
)
from predictors.convolution import ConvolutionPredictor
from predictors.graph_boosting_tree import (
    GraphBoostingTreePredictor,
    GraphBoostingTreePredictor2,
    GraphBoostingTreePredictor3,
)
from predictors.decision_tree import AugmentedPredictor
from predictors.subpattern import SubpatternMatcherPredictor
from predictors.connector import PointConnectorPredictor
from predictors.cf_combinator import WrappedCFPredictor
