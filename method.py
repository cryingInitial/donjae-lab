from copy import deepcopy
from methods.pretrain import Pretrain
from methods.FT import FT
from methods.neggrad import NegGrad, NegGradP
from methods.teacher import Teacher
from methods.boundary_expand import BoundaryExpand
from methods.boundary_shrink import BoundaryShrink
from methods.cfk import CF_k
from methods.euk import EU_k
from methods.ntk import NTK
from methods.scrub import SCRUB
from methods.scrub_r import SCRUB_R
from methods.LP import LP
from methods.LPFT import LPFT
# from methods.fisher import Fisher
from methods.lau import LAU
from methods.sparse import SPARSE
from methods.distill import Distill
# from methods.distill_LPFT import DistillLPFT

def run_method(model, retrain_model, loaders, args):
    model = deepcopy(model) # to ensure the original model is not modified
    globals_lower = {k.lower(): v for k, v in globals().items()}
    return globals_lower[args.method.lower()](model, retrain_model, loaders, args).run()
