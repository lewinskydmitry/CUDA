#include "MatrixOperations.cuh"

__global__
void cudaAddMatrixKernel(const double* a,
    const double* b,
    double* c,
    const int size) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_idx < size) {
        c[thread_idx] = a[thread_idx] + b[thread_idx];
        thread_idx += blockDim.x * gridDim.x;
    }
}


Matrix AddMatrix(Matrix a, Matrix b) {
    int size;
    Matrix c;
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    int per_block_thread_count = 1024;

    if (a.length != b.length && a.width != b.width) {
        throw std::invalid_argument("Shapes aren't matching");
    }
    else {
        size = a.length * a.width;
        c.length = a.length;
        c.width = a.width;
        c.data = new double[size];
    };

    int block_count = (int)ceil(size / (float)per_block_thread_count);


    cudaMalloc((void**)&dev_c, size * sizeof(double));
    cudaMalloc((void**)&dev_a, size * sizeof(double));
    cudaMalloc((void**)&dev_b, size * sizeof(double));


    cudaMemcpy(dev_a, a.data, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data, size * sizeof(double), cudaMemcpyHostToDevice);

    cudaAddMatrixKernel << < block_count, per_block_thread_count >> > (dev_a, dev_b, dev_c, size);

    cudaDeviceSynchronize();

    cudaMemcpy(c.data, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);


    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return c;
};