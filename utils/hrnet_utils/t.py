import numpy as np
import torch


x = np.array([1,2,3,4,5,6,7,8,9,12])

y = 1

z = np.array([1,2,3,4,5,6,7,8,9,12])

y += x

y += z
area_target, _ = np.histogram(y, bins=np.arange(30+1))
area_output = torch.histc(torch.from_numpy(y).float().cpu(), bins=30, min=0, max=30-1)
print(y)
print(area_target)
print(area_output.cpu().int().numpy())

a = torch.randn(1, 25405)
a = a.max(1)[1]
print(a)
