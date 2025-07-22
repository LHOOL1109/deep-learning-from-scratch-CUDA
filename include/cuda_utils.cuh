#pragma once

#define IDX_1D (threadIdx.x + blockDim.x * blockIdx.x)
#define RETURN_IF_OOB(idx, n) if ((idx) >= (n)) return;