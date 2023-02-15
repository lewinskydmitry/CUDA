#include "MatrixOperations.cuh"


__global__ void TransposeKernel(Matrix idata, Matrix odata) {
    __shared__ double tile[BLOCK_SIZE][BLOCK_SIZE];
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;


    int i;
    if (x < idata.width && (y + i) < idata.length) {
        tile[threadIdx.y + i][threadIdx.x] = idata.data[(y + i) * idata.width + x];
    }


    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (x < idata.length && (y + i) < idata.width) {
        odata.data[(y + i) * idata.length + x] = tile[threadIdx.x][threadIdx.y + i];

    }
}


Matrix Transpose(Matrix A)
{
    Matrix d_A;
    d_A.width = A.width; d_A.length = A.length;
    size_t size = A.width * A.length;
    cudaMalloc(&d_A.data, size * sizeof(double));
    cudaMemcpy(d_A.data, A.data, size * sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.length = A.width;
    d_C.width = A.length;
    cudaMalloc(&d_C.data, size * sizeof(double));

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    dim3 dimGrid((A.width + dimBlock.x - 1) / dimBlock.x, (A.length + dimBlock.y - 1) / dimBlock.y);
    TransposeKernel <<< dimGrid, dimBlock >>> (d_A, d_C);

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