import math
import torch

DIMENSION = 1
N_HIDDEN = 1

def erf_combo(x):
    return torch.erf(x) - 2.0 * torch.erf(0.5 * x)

# Let's find MSE using w = -0.2444508388
w = -0.2444508388

# A(a,b)
def A(a, b):
    return (2/math.pi) * math.asin((2*a*b)/(math.sqrt(1+2*a**2)*math.sqrt(1+2*b**2)))

mse = 0.5 * ( A(w,w) + A(1,1) + 4*A(0.5,0.5) - 2*A(w,1) + 4*A(w,0.5) - 4*A(1,0.5) )
print(f"Theoretical MSE for optimal w: {mse:.8f}")

