"""
" heat.py
"
" This file contains implementations to solve one-dimensional heat transfer
" equation on CPU and with GPU acceleration using PyCUDA.
"
:Copyright: Mikko Majamaa
:Author: Mikko Majamaa
:Date: 21 June, 2020
:Version: 1.0.0
"""

# built-in package imports
import time
# third party package imports
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit


def heat_cpu(u, M, D, T, N):
    """
    Solve heat transfer equation using CPU.

    @param u: (np.ndarray) Initial heat distribution.
    @param M: (int) Number of temporal layers.
    @param D: (float) Diffusion co-efficient.
    @param T: (int) Integration time.
    @param N: (int) Number of spatial nodes.
    @return (np.ndarray) Final heat distribution.
    """    

    dx = 1 / (N - 1)
    dt = T / (M - 1)

    left = np.concatenate(([u[0]], u[0:N-1]))
    right = np.concatenate((u[1:N], [u[N-1]]))

    for j in range(M):
        if (j % 100000 == 0):
            print('Starting step {}'.format(j))

        left = np.concatenate(([u[0]], u[0:N-1]))
        right = np.concatenate((u[1:N], [u[N-1]]))

        u += D * dt / (dx * dx) * (left - 2 * u + right)
    
    return u


def heat_gpu(u, M, D, T, N):
    """

    Function responsible of handling solving the heat transfer equation on GPU.

    @param u: (np.ndarray) Initial heat distribution.
    @param M: (int) Number of temporal layers.
    @param D: (float) Diffusion co-efficient.
    @param T: (int) Integration time.
    @param N: (int) Number of spatial nodes.
    @return (np.ndarray) Final heat distribution.
    """    

    mod = SourceModule("""
    __global__ void heat(
        double *u, // initial heat distribution
        int M, // number of temporal layers
        double D, // diffusion co-efficient
        double T, // integration time
        int N // number of spatial nodes
        ) 
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;

        //double D = 0.005;
        if (x >= N) {
            return;
        }
        
        // Transfer data to a shared array.
        extern __shared__ double sdata[];
        sdata[x] = u[x];
        __syncthreads();
        
        double dt = T/(M-1); // time step
        double dx = 1.0/(N-1); // spatial step

        double left; // node's left neighbour
        double right; // node's right neighbour

        for (int i = 0; i < M; i++) {
            if (x == 0)
                left = sdata[0];
            else
                left = sdata[x - 1];
            if (x == N - 1)
                right = sdata[N - 1];
            else
                right = sdata[x + 1];

            sdata[x] += D * dt / (dx * dx) * (left - 2 * sdata[x] + right);
        
            __syncthreads();
        }

        u[x] = sdata[x];
    }
    """)

    BLOCKSIZE = (N, 1, 1)
    GRIDSIZE = (1, 1, 1)

    func = mod.get_function('heat')
    u_gpu = cuda.mem_alloc(u.nbytes)
    cuda.memcpy_htod(u_gpu, u)

    func(u_gpu, np.int32(M), np.float64(D), np.float64(T), np.int32(N), block=BLOCKSIZE, grid=GRIDSIZE, shared=u.nbytes)
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(u, u_gpu)

    return u


def main(M, D, T, N):
    """
    Main function responsible of calling the needed functions.

    @param M: (int) Number of temporal layers.
    @param D: (float) Diffusion co-efficient.
    @param T: (int) Integration time.
    @param N: (int) Number of spatial nodes.
    """    

    u_cpu = np.random.rand(N)  # initial temperature distribution
    u_gpu = np.array(u_cpu, copy=True)

    # start of running on CPU
    cpu_start = time.perf_counter()
    u_cpu = heat_cpu(u_cpu, M, D, T, N)
    cpu_end = time.perf_counter()

    # start of running on GPU (shared memory solution)
    gpu_start = time.perf_counter()
    u_gpu = heat_gpu(u_gpu, M, D, T, N)
    gpu_end = time.perf_counter()

    if (np.allclose(u_cpu, u_gpu, rtol=1e-03, atol=1e-06)):
        print('Computation took {}s on CPU.'.format(cpu_end - cpu_start))
        print('Computation took {}s on GPU.'.format(gpu_end - gpu_start))
    else:
        print(u_cpu[1000:1023])
        print(u_gpu[1000:1023])
        print(sum(np.abs(u_cpu)-np.abs(u_gpu)))
        print('Results do not match!')


if __name__ == '__main__':
    M = 10**6 # number of temporal layers
    D = 0.005 # diffusion co-efficient
    T = 50 # integration time
    N = 1024 # number of spatial nodes (must be < 1024 because of the use of shared memory)

    main(M, D, T, N)