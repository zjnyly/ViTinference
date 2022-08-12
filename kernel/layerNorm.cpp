#pragma once

// x = torch.rand(2,3,4,5)
// layer = nn.LayerNorm(5)

// mean = x.mean(axis=3).reshape(-1,x.shape[1],x.shape[2],1)
// var = x.var(axis=3,unbiased=False).reshape(-1,x.shape[1],x.shape[2],1)
// out2 = (x-mean)/((var+1e-5)**0.5)
