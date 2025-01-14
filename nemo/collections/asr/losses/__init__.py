# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo.collections.asr.losses.angularloss import AngularSoftmaxLoss
from nemo.collections.asr.losses.bce_loss import BCELoss
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.lattice_losses import LatticeLoss
from nemo.collections.asr.losses.ssl_losses.contrastive import ContrastiveLoss
from nemo.collections.asr.losses.ssl_losses.ctc import CTCLossForSSL
from nemo.collections.asr.losses.ssl_losses.mlm import MLMLoss, MultiMLMLoss
from nemo.collections.asr.losses.ssl_losses.rnnt import RNNTLossForSSL
