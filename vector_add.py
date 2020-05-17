"""
" vector_add.py
"
" This file contains vector addition implementations on CPU only
" and with GPU acceleration using PyCUDA.
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


def main(size, data_type):
    """
    Main function responsible of calling the needed functions.
    
    @param size: (int) Length of the vectors to be added.
    @param data_type: (list[str]) Data types with which to run the vector addition.
    """

    for data_type in data_types:
        # get the types
        (np_type, c_type) = d_types(data_type)

        # define the input vectors (a and b) and the output vector (c) 
        a = numpy.random.randn(size, 1)
        a = a.astype(np_type)
        b = numpy.random.randn(size, 1)
        b = b.astype(np_type)
        c = numpy.zeros(shape=(size, 1))
        c = c.astype(np_type)

        # start of the GPU implementation
        start_gpu = time.perf_counter()
        c_res_gpu = vector_add_gpu(a, b, c, size, c_type)
        end_gpu = time.perf_counter()
        print("Vector add on GPU took {}ms ({})".format((end_gpu - start_gpu) * 1000, data_type))
        
        # Start of the CPU implementation
        start_cpu = time.perf_counter()
        c_res_cpu = vector_add_cpu(a, b, c, size)
        end_cpu = time.perf_counter()
        print("Vector add on CPU took {}ms ({})".format((end_cpu - start_cpu) * 1000, data_type))

        if (numpy.allclose(c_res_gpu, c_res_cpu)):
            print('Results match!')
        else:
            print('Results differ!')


def vector_add_gpu(a, b, c, size, d_type):
    """
    Function responsible for invoking vector addition on GPU.

    @param a: (np.ndarray) Input vector.
    @param b: (np.ndarray) Input vector.
    @param c: (np.ndarray) Output vector.
    @param size: (int) Length of the vectors to be added.
    @param d_type: (str) Data type's name of the vector elements. 
    @return (np.ndarray) Output vector.
    """    

    # 1-dimensional block with length of 1024
    BLOCKSIZE = (1024, 1, 1)
    # 1-dimensional grid with length so that each element of the vectors are added on a separate thread
    # "size//BLOCKSIZE[0]+1" makes sure that every element is computed
    GRIDSIZE = (size // BLOCKSIZE[0] + 1, 1, 1)

    # allocate memory on the device and transfer the data
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    b_gpu = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(b_gpu, b)
    c_gpu = cuda.mem_alloc(c.nbytes)
    cuda.memcpy_htod(c_gpu, c)

    # get the template and replace the placeholders
    tpl = Template(kernel_code)
    rendered_tpl = tpl.render(type_name=d_type)
    mod = SourceModule(rendered_tpl)

    # run the vector addition
    func = mod.get_function("vector_add")
    func(a_gpu, b_gpu, c_gpu, numpy.int32(size), block=BLOCKSIZE, grid=GRIDSIZE)

    # transfer the data back to host
    c_res_gpu = numpy.empty_like(c)
    cuda.memcpy_dtoh(c_res_gpu, c_gpu)

    return c


kernel_code = """
    __global__ void vector_add(
        {{type_name}} *a, // input vector
        {{type_name}} *b, // input vector
        {{type_name}} *c, // output vector
        int size
        )
{
        // get the global index of the thread
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // check that the thread is not outside of the bounds of the computation
        if (idx >= size) return;

        c[idx] = a[idx] + b[idx];
}
"""


def vector_add_cpu(a, b, c, size):
    """
    Function responsible for handling vector addition on CPU.

    @param a: (np.ndarray) Input vector.
    @param b: (np.ndarray) Input vector.
    @param c: (np.ndarray) Output vector.
    @param size: (int) Length of the vectors to be added.
    @return (np.ndarray) Output vector.
    """    

    c_res_cpu = numpy.empty_like(c)

    vectorized_add_elements = numpy.vectorize(add_elements)

    c_res_cpu = vectorized_add_elements(a, b)

    return c


def add_elements(a, b):
    """
    Helper function to add elements a and b.        
    """
    return a + b


if __name__ == '__main__':
    # length of the vectors to be added
    size = 1000**2
    
    # different data types with which to run the vector addition
    data_types = ['float', 'double', 'int']
    
    main(size, data_types)