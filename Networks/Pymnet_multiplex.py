import pymnet
import numpy as np
import matplotlib.pyplot as plt
from pymnet import *

fig=draw(er(15,2*[0.3]),
             layout="spring",
             layershape="rectangle",
             nodeColorDict={(0,0):"black",(1,0):"black",(0,1):"black"},
             layerLabelRule={},
             nodeLabelRule={},
             nodeSizeRule={"rule":"degree","propscale":0.05},
             show=True)

import pymnet
import os

# Get the path of the pymnet module
pymnet_path = os.path.dirname(pymnet.__file__)

print("pymnet is installed at:", pymnet_path)