#include "MatrixOperations.cuh"


__global__ void TransposeKernelRep(Matrix idata) {
    __shared__ double tile[BLOCK_SIZE][BLOCK_SIZE];
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y; 
    int i;

    for (int i = 0; i < BLOCK_SIZE; i += blockDim.y) {
        if (x < idata.width && (y + i) < idata.length) {
            tile[threadIdx.y + i][threadIdx.x] = idata.data[(y + i) * idata.width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    for (int i = 0; i < BLOCK_SIZE; i += blockDim.y) {
        if (x < idata.length && (y + i) < idata.width) {
            idata.data[(y + i) * idata.length + x] = tile[threadIdx.x][threadIdx.y + i];

        }
    }
}


void TransposeRep(Matrix& A)
{
    Matrix d_A;
    d_A.width = A.width; d_A.length = A.length;
    size_t size = A.width * A.length;
    cudaMalloc(&d_A.data, size * sizeof(double));
    cudaMemcpy(d_A.data, A.data, size * sizeof(double), cudaMemcpyHostToDevice);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((A.width + dimBlock.x - 1) / dimBlock.x, (A.length + dimBlock.y - 1) / dimBlock.y);
    TransposeKernelRep <<< dimGrid, dimBlock >>> (d_A);

    cudaMemcpy(A.data, d_A.data, size * sizeof(double), cudaMemcpyDeviceToHost);

    A.length = d_A.width;
    A.width = d_A.length;

    cudaFree(d_A.data);
}