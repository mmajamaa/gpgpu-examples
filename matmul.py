"""
" vector_add.py
"
" This file contains matrix multiplication implementations on CPU only
" using Numpy's built-in function (to check the correctness of the GPU
" implementation) and with GPU acceleration using PyCUDA.
"
:Copyright: Mikko Majamaa
:Author: Mikko Majamaa
:Date: 17 May, 2020
:Version: 1.0.0
"""

# built-in package imports
import time
# third party package imports
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy
from jinja2 import Template
# own module imports
from helpers import d_types


def main(n11, n12, n21, n22, data_types):
    """
    Main function responsible of calling the needed functions.

    @param n11: (int) Rows in the first input matrix.
    @param n12: (int) Columns in the first input matrix.
    @param n21: (int) Rows in the second input matrix.
    @param n22: (int) Columns in the second input matrix.
    @return (np.ndarray) Output matrix.@param data_types: (list[str]) Data types with which to run the vector addition.
    """    

    for data_type in data_types:
        # get the types
        (np_type, c_type) = d_types(data_type)

        # define the input and output matrices
        a = numpy.random.rand(n11, n12)
        a = a.astype(np_type)
        b = numpy.random.rand(n21, n22)
        b = b.astype(np_type)
        c = numpy.zeros(shape=(n11, n22))
        c = c.astype(np_type)

        # start of the GPU implementation
        start_gpu = time.perf_counter()
        c_res_gpu = matmul_gpu(a, b, c, n11, n12, n21, n22, c_type)
        end_gpu = time.perf_counter()
        print("Matrix multiplication on GPU took {}ms ({})".format((end_gpu - start_gpu) * 1000, data_type))

        # start of the matrix multiplication using Numpy
        start_cpu = time.perf_counter()
        c_res_cpu = numpy.matmul(a, b)
        end_cpu = time.perf_counter()
        print("Matrix multiplication on CPU took {}ms ({})".format((end_cpu - start_cpu) * 1000, data_type))

        # confirm correctness of the results
        if (numpy.allclose(c_res_gpu, c_res_cpu)):
            print('Results match!')
        else:
            print('Results differ!')


def matmul_gpu(a, b, c, n11, n12, n21, n22, data_type):
    """
    Function responsible for invoking matrix multiplication on GPU.

    @param a: (np.ndarray) Input matrix.
    @param b: (np.ndarray) Input matrix.
    @param c: (np.ndarray) Output matrix.
    @param n11: (int) Rows in the first input matrix.
    @param n12: (int) Columns in the first input matrix.
    @param n21: (int) Rows in the second input matrix.
    @param n22: (int) Columns in the second input matrix.
    @param data_type: (str) Data type's name of the matrix elements.
    @return (np.ndarray) Output matrix.
    """    

    # define the block and the grid sizes
    BLOCKSIZE = (32, 32, 1)
    GRIDSIZE = (n22 // BLOCKSIZE[0] + 1, n11 // BLOCKSIZE[1] + 1)

    # allocate memory and transfer the data
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    b_gpu = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(b_gpu, b)
    c_gpu = cuda.mem_alloc(c.nbytes)
    cuda.memcpy_htod(c_gpu, c)

    # get the template and replace the placeholders
    tpl = Template(kernel_code)
    rendered_tpl = tpl.render(data_type=data_type)
    mod = SourceModule(rendered_tpl)

    # run the matrix multiplication
    func = mod.get_function("matmul")
    func(a_gpu, b_gpu, c_gpu, numpy.int32(n11), numpy.int32(n12), numpy.int32(n21), numpy.int32(n22), block=BLOCKSIZE, grid=GRIDSIZE)

    # transfer the data back to host
    c_res_gpu = numpy.empty_like(c)
    cuda.memcpy_dtoh(c_res_gpu, c_gpu)

    return c_res_gpu



kernel_code = """
__global__ void matmul(
        {{data_type}} *a, // input matrix
        {{data_type}} *b, // input matrix
        {{data_type}} *c, // output matrix
        int n11, // rows in the first input matrix
        int n12, // columns in the first input matrix
        int n21, // rows in the second input matrix
        int n22 // columns in the second input matrix
    )
{
    // get the global indices of the thread
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    // check that the thread is not outside of the bounds of the computation
    if ((x >= n22) || (y >= n11)) { // out of c matrix bounds
        return;
    }
    
    {{data_type}} res = 0; 
    
    for (int i = 0; i < n12; i++) {
        {{data_type}} ael = a[y*n12 + i];
        {{data_type}} bel = b[i*n22 + x];
        res += ael * bel;
    }
    
    c[y*n22+x] = res;
}
"""


if __name__ == '__main__':
    # dimensions for the first input matrix
    n11 = 10**4
    n12 = 3000

    # dimensions for the second input matrix
    n21 = n12
    n22 = 10**3

    # different data types with which to run the vector addition
    data_types = ['double', 'float', 'int']
    
    main(n11, n12, n21, n22, data_types)