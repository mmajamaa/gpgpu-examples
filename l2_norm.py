"""
" l2_norm.py
"
" This file contains computing L2 norm with GPU acceleration using PyCUDA,
" as well as a performance comparison against it's equivalent on Numpy.
"
:Copyright: Mikko Majamaa
:Author: Mikko Majamaa
:Date: 18 July, 2020
:Version: 1.0.0
"""


# built-in package imports
import time
# third party package imports
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit


d = 1 # increase this to increase total size of N
f = 1024 * d
N = 1024 * f
BLOCKSIZE = (1024, 1, 1)
GRIDSIZE = (1 * f, 1, 1)

mod = SourceModule("""
    __device__ int warpReduce(int val) {
        // first warp reduce that computes square of each element
        int temp;
        for (int offset = warpSize/2; offset > 0; offset /=2) {
            if (offset == warpSize/2) {
                val *= val;
            }
            temp = __shfl_down_sync(0xFFFFFFFF, val, offset);
            val += temp;
        }
        return val;
    }

    __device__ int warpReduce2(int val) {
        // second warp reduce to add the values of the elements' squares
        int temp;
        for (int offset = warpSize/2; offset > 0; offset /=2) {
            temp = __shfl_down_sync(0xFFFFFFFF, val, offset);
            val += temp;
        }
        return val;
    }

    __device__ int blockReduce(int val) {
        static __shared__ int shared[32]; // results from reducing the block
        int lane = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize; // warp id

        val = warpReduce(val); // reduce values within warps

        if (lane == 0) { // reduce results from different warps (earch warp's result is in lane 0)
            shared[wid] = val;
        }

        __syncthreads();

        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

        if (wid == 0) {
            val = warpReduce2(val); // values from the block are in warp 0 so reduce that warp
        }

        return val;
    }
    __global__ void l2norm(int *x, int* res) {
        // main device function
        int tx = threadIdx.x + blockIdx.x * blockDim.x;
        int bx = blockIdx.x;

        int temp = 0;
        temp = blockReduce(x[tx]);
        if (threadIdx.x == 0) { // values from each warp is in thread 0 so add it to final result
            atomicAdd(res, temp);
        }
    }
    """)

xv = np.random.randint(10, size=1024 * f, dtype=np.int32)
res = np.empty(1024*d, dtype=np.int32)
res = np.array(0)

start_gpu = time.perf_counter()
func = mod.get_function('l2norm')
xv_gpu = cuda.mem_alloc(xv.nbytes)
res_gpu = cuda.mem_alloc(res.nbytes)

cuda.memcpy_htod(xv_gpu, xv)
cuda.memcpy_htod(res_gpu, res)

func(xv_gpu, res_gpu, block=BLOCKSIZE, grid=GRIDSIZE)

res_gpu_r = np.empty_like(res)
cuda.memcpy_dtoh(res_gpu_r, res_gpu)
end_gpu = time.perf_counter()

start_cpu = time.perf_counter()
res_cpu = np.linalg.norm(xv)
end_cpu = time.perf_counter()


if np.allclose(res_cpu, np.sqrt(res_gpu_r)):
    print('Results match!')
    print('Computing L2 norm took {}s on GPU'.format(end_gpu - start_gpu))
    print('Computing L2 norm took {}s on CPU'.format(end_cpu - start_gpu))
else:
    print('Results differ!')
