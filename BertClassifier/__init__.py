

import torch.nn as nn
import torch


if __name__ == '__main__':
    m = nn.ZeroPad2d(2)
    input = torch.randn(1, 1, 3, 3)
    print(input)
    a=m(input)
    print(a,a.shape)
    m2=nn.ZeroPad2d((2,3,1,1))
    b=m2(input)
    print(b,b.shape)