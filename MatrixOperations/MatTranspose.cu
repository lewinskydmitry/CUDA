#include "MatrixOperations.cuh"

__global__ void copySharedMem(Matrix odata, const Matrix idata)
{
    __shared__ double tile[BLOCK_SIZE * BLOCK_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    for (int j = 0; j < BLOCK_SIZE; j += BLOCK_SIZE)
        tile[(threadIdx.y + j) * BLOCK_SIZE + threadIdx.x] = idata.data[(y + j) * idata.width + x];

    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; j += BLOCK_SIZE)
        odata.data[(y + j) * odata.width + x] = tile[(threadIdx.x + j) * BLOCK_SIZE + threadIdx.y];
}


Matrix MatTranspose(Matrix A)
{
    Matrix d_A;
    d_A.width = A.width; d_A.length = A.length;
    size_t size = A.width * A.length;
    cudaMalloc(&d_A.data, size * sizeof(double));
    cudaMemcpy(d_A.data, A.data, size * sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.length = A.width;
    d_C.width = A.length;
    size = A.length * A.width;
    cudaMalloc(&d_C.data, size * sizeof(double));

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((A.width + dimBlock.x - 1) / dimBlock.x, (A.length + dimBlock.y - 1) / dimBlock.y);
    copySharedMem << <dimGrid, dimBlock >> > (d_C, d_A);

    Matrix C;
    C.length = A.width;
    C.width = A.length;
    C.data = new double[size];

    // Read C from device memory
    cudaMemcpy(C.data, d_C.data, size * sizeof(double), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.data);
    cudaFree(d_C.data);
    return C;
}