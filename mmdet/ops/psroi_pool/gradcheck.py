import os.path as osp
import sys

import torch
from torch.autograd import gradcheck

sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from mmdet.ops import PSRoIPool  # noqa: E402, isort:skip

feat = torch.randn(4, 16, 15, 15, dtype=torch.float32, requires_grad=True).cuda()
rois = torch.as_tensor(
        torch.Tensor([[0, 0, 0, 50, 50], [0, 10, 30, 43, 55],
                     [1, 67, 40, 110, 120]]), dtype=torch.float32).cuda()
inputs = (feat, rois)
print('Gradcheck for roi pooling...')
test = gradcheck(PSRoIPool(4, 4, 1.0 / 8, 4, 9), inputs, eps=1e-5, atol=1e-3)
print(test)
