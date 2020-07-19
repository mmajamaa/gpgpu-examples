"""
" mean_filter.py
"
" This file contains mean filter implementations using only CPU
" and GPU acceleration and their comparison in execution time.
"
:Copyright: Mikko Majamaa
:Author: Mikko Majamaa
:Date: 19 July, 2020
:Version: 1.0.0
"""


# built-in package imports
from time import perf_counter
import os
# third party package imports
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import matplotlib.pyplot as plt
from jinja2 import Template


imageName = 'test1' # input image's name (test1/test2/test3)
L = 10 # window size
image = plt.imread(os.path.join('assets', imageName) + '.png')
imageDims = image.shape

# define vectors to store result
image_final_gpu = np.empty_like(image)
image_final_cpu = np.empty_like(image)

kernelCode = """
__global__ void meanFilter(
    float *image, // input image as a vector
    float *J, // output image
    int L, // window size
    int i, // rows in the original image
    int j // columns in the original image
    )
{
    // Each thread is responsible for computing the result of a
    // pixel in J.

    // get global coordinates of the thread
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // check that coordinates are not out of bounds of the computation
    if ((x >= j) | (y >= i)) return;

    // variable into which J's element is temporarily stored
    float temp = 0;

    for (int k = y-L; k <= y+L; k++) {
        // pixels that are out of bounds are 0
        if ((k < 0) | (k >= i)) continue;
        for (int l = x-L; l <= x+L; l++) {
            // pixels that are out of bounds are 0
            if ((l < 0) | (l >= j)) continue;
            temp += image[k*j+l];
        }
    }

    float temp_L = (float) L;
    J[y*j+x] = 1/((2*temp_L+1)*(2*temp_L+1))*temp;
}
"""

tpl = Template(kernelCode)
rendered_tpl = tpl.render()
mod = SourceModule(rendered_tpl)

func = mod.get_function("meanFilter")

# define the grid
blockSize = 32
block = (blockSize, blockSize, 1)
grid = (imageDims[0]//blockSize+1, imageDims[1]//blockSize+1, 1)

# allocate memory and transfer data
gpu_start = perf_counter()
image_gpu = cuda.mem_alloc(image.nbytes)
output_image_gpu = cuda.mem_alloc(image.nbytes)
cuda.memcpy_htod(image_gpu, image)

# integrate model on GPU
func(image_gpu, output_image_gpu, np.int32(L), np.int32(imageDims[0]), np.int32(imageDims[1]), block=block, grid=grid)
cuda.Context.synchronize()

# transfer result back to host
cuda.memcpy_dtoh(image_final_gpu, output_image_gpu)
gpu_end = perf_counter()
plt.imsave(imageName + '_filtered_gpu.png', image_final_gpu)

# integrate model on CPU
image_cpu = np.copy(image)
cpu_start = perf_counter()

# iterate through rows of the original picture
for i in range(0, imageDims[0]):
    # iterate through columns of the original picture
    for j in range(0, imageDims[1]):
        # variable into which J's element is temporarily stored
        temp = 0    
        for k in range(i-L, i+L+1):
            # pixels that are out of bounds are 0
            if (k < 0) | (k >= imageDims[0]):
                continue
            for l in range(j-L, j+L+1):
                # pixels that are out of bounds are 0
                if (l < 0) | (l >= imageDims[1]):
                    continue
                temp += image[k,l]
        image_final_cpu[i,j] = 1/(2*L+1)**2*temp
cpu_end = perf_counter()

plt.imsave(imageName + '_filtered_cpu.png', image_final_cpu)

# print output info
if np.allclose(image_final_gpu, image_final_cpu):
    print('Final images match!')
    print('Computation took {}ms on GPU'.format(gpu_end-gpu_start))
    print('Computation took {}ms on CPU'.format(cpu_end-cpu_start))
else:
    print('Final images do not match!')
