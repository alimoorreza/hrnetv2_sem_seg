from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .default import _C as config
from .default_hrnet import _C as config_hrnet
from .default import update_config
from .default_hrnet import update_config as update_config_hrnet
from .models_hrnet import MODEL_EXTRAS

from .default_hrnet_v2 import _C as config_hrnet_v2
from .default_hrnet_v2 import update_config as update_config_hrnet_v2
from .models_hrnet_v2 import MODEL_EXTRAS as MODEL_EXTRAS_HRNET_V2

"""import sys
sys.path.append("...")"""