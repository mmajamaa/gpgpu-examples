"""
" mean_filter.py
"
" This file contains mean filter implementations using only CPU
" and GPU acceleration (with multiple GPU devices) and their comparison
" in execution time.
"
:Copyright: Mikko Majamaa
:Author: Mikko Majamaa
:Date: 23 July, 2020
:Version: 1.0.0
"""


# built-in package imports
from time import perf_counter
# third party package imports
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import matplotlib.pyplot as plt
from jinja2 import Template


imageName = 'test3' # input image's name (test1/test2/test3)
L = 5 # window size
image = plt.imread(imageName + '.png')
imageDims = image.shape

# define matrices to store result
image_final_gpu = np.empty_like(image)
image_final_cpu = np.empty_like(image)

gpu_start = perf_counter()

# init CUDA
cuda.init()
# define CUDA objects for each available device
device = [] # CUDA devices
context = [] # CUDA contexts
stream = [] # CUDA streams
i_dev = [] # device allocations for the parts of matrix I
j_dev = [] # device allocations for the parts of matrix J

# init device array
device.append(cuda.Device(0))
device.append(cuda.Device(1))

# rows in the original image
n_rows = image.shape[0]
# GPU device count
n_devs = len(device)

# init contexts
for i in range(0, n_devs):
    context.append(device[i].make_context())

# init streams
for i in range(0, n_devs):
    # activate context for current device
    context[i].push()
    # create streams
    stream.append(cuda.Stream())
    # deactivate context for current device
    context[i].pop()

# activate the first context and register memory on host
context[0].push()
# define matrices in which each dev will compute slice of the final matrix
j_regs = []
i_reg = cuda.register_host_memory(image, 1)
for dev in device:
    # each slice that will contain the partial matrix of the final image
    # need to be L rows greater than the fraction of n_rows/n_devs to correctly
    # compute elements at the edges
    image_final_gpu_slice = np.empty(shape=(imageDims[0]//n_devs+L, imageDims[1]), dtype=np.float32)
    j_reg = cuda.register_host_memory(image_final_gpu_slice, 1)
    j_regs.append(j_reg)
# "here flag 1 stands for cudaHostRegisterPortable, which means that the pinned memory will be available
# for all contexts at the same time"
context[0].pop()

# split the data in array 'image' to equal parts
# and transfer the data to GPUs asynchronously
for i in range(0, n_devs):
    # input matrix is divided in vertical direction
    row_start = (i > 0)*(i*n_rows//n_devs - L)
    
    row_end = (i+1)*(n_rows//n_devs + L)\
     if (i < n_devs - 1)\
     else n_rows
    
    slice_i = i_reg[row_start:row_end, :]
    slice_j = j_regs[i]

    # activate context for current device
    context[i].push()

    # allocate memory
    i_dev.append(cuda.mem_alloc(slice_i.nbytes))
    j_dev.append(cuda.mem_alloc(slice_j.nbytes))

    # copy data
    cuda.memcpy_htod_async(i_dev[i], slice_i, stream[i])
    cuda.memcpy_htod_async(j_dev[i], slice_j, stream[i])

    # deactivate context for current device
    context[i].pop()


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


for i in range(0, n_devs):
    context[i].push()

    tpl = Template(kernelCode)
    rendered_tpl = tpl.render()
    mod = SourceModule(rendered_tpl)

    func = mod.get_function("meanFilter")

    # define the grid
    blockSize = 32
    block = (blockSize, blockSize, 1)
    grid = (imageDims[0]//blockSize+1+L, imageDims[1]//blockSize+1, 1)

    func(i_dev[i], j_dev[i], np.int32(L), np.int32(imageDims[0]/n_devs+L), np.int32(imageDims[1]), block=block, grid=grid)

    context[i].pop()

# transfer data back to host
for i in range(0, n_devs):
    row_start = i*image.shape[0]//n_devs
    row_end = (i+1)*image.shape[0]//n_devs 
    # extract host array views
    slice_j = j_regs[i]
    # activate context
    context[i].push()
    # copy the data from current device to host
    cuda.memcpy_dtoh_async(slice_j, j_dev[i], stream[i])
    # deactivate context
    print(slice_j)
    context[i].pop()

# sync streams
for i in range(0, n_devs):
    stream[i].synchronize()

# form the final image
for i in range(0, n_devs):
    i_row_start = i*image.shape[0]//n_devs
    i_row_end = (i+1)*image.shape[0]//n_devs
    # slices of the final image have overlapping elements, so different
    # indexing to select elements from them than in the matrix that will
    # contain the whole image is needed
    j_row_start = 0 if i == 0 else L
    j_row_end = image.shape[0]//n_devs if i == 0 else i*image.shape[0]//n_devs + L
    image_final_gpu[i_row_start:i_row_end, :] = j_regs[i][j_row_start:j_row_end, :]




# unregister memory by using the same context which registered it
context[0].push()
i_reg.base.unregister()
for i in range(0, n_devs):
    j_regs[i].base.unregister()
context[0].pop()

# manually destroy all CUDA contexts
for i in range(0, len(device)):
    cuda.Context.pop()

gpu_end = perf_counter()

# integrate model on CPU
cpu_start = perf_counter()
image_cpu = np.copy(image)

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
plt.imsave(imageName + '_filtered_gpu.png', image_final_gpu)

# print output info
if np.allclose(image_final_gpu, image_final_cpu):
    print('Final images match!')
    print('Computation took {}ms on GPU'.format(gpu_end-gpu_start))
    print('Computation took {}ms on CPU'.format(cpu_end-cpu_start))
else:
    print('Final images do not match!')
