#include "LinearRegression.cuh"


__global__ void bias_update(Matrix difference, Matrix bias, float THETA)
{
    int size = difference.width * difference.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_idx < size) {
        double result = 0;
        for (int i = 0; i < size; i++) {
            result += difference.data[thread_idx] / difference.length;
        }
        bias.data[thread_idx] = THETA * 2 * result;
        thread_idx += blockDim.x * gridDim.x;
    }
}


__global__ void weights_update(Matrix weights, Matrix weightsGrad, float THETA)
{
    int size = weightsGrad.width * weightsGrad.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_idx < size) {
        weights.data[thread_idx] += THETA * 2 * weightsGrad.data[thread_idx] / weightsGrad.length;
        thread_idx += blockDim.x * gridDim.x;
    }
}


__global__ void LossFunc(Matrix erMatrix, Matrix difference, int iteration) {
    int size = difference.width * difference.length;
    float error = 0;
    for (int e = 0; e < size; e++) {
        error += pow(difference.data[e], 2) / difference.length;
    }
    erMatrix.data[iteration] = error;
}


Matrix LinearRegression::fit(Matrix X, Matrix y, int epochs) {

    // INIT MATRICES
    Matrix d_X;
    d_X.width = X.width; d_X.length = X.length; d_X.stride = X.width;
    cudaMalloc(&d_X.data, X.width * X.length * sizeof(double));
    cudaMemcpy(d_X.data, X.data, X.width * X.length * sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_y;
    d_y.width = y.width; d_y.length = y.length;  d_y.stride = y.width;
    cudaMalloc(&d_y.data, y.width * y.length * sizeof(double));
    cudaMemcpy(d_y.data, y.data, y.width * y.length * sizeof(double), cudaMemcpyHostToDevice);

    // VARIABLES FOR OPTIMIZATION
    Matrix d_w;
    d_w.width = y.width; d_w.length = X.width; d_w.stride = y.width;
    cudaMalloc(&d_w.data, X.length * sizeof(double));

    Matrix d_wGrad;
    d_wGrad.width = y.width; d_wGrad.length = X.width; d_wGrad.stride = y.width;
    cudaMalloc(&d_wGrad.data, X.length * sizeof(double));

    Matrix d_b;
    d_w.width = y.width; d_w.length = y.length; d_w.stride = y.width;
    cudaMalloc(&d_b.data, y.width * y.length * sizeof(double));

    // VARIABLES FOR COMPUTING
    Matrix d_pred;
    d_pred.width = y.width; d_pred.length = y.length; d_pred.stride = y.width;
    cudaMalloc(&d_pred.data, y.width * y.length * sizeof(double));

    Matrix d_difference;
    d_difference.width = y.width; 
    d_difference.length = y.length;
    d_difference.stride = y.width;
    cudaMalloc(&d_difference.data, y.width * y.length * sizeof(double));

    // ERROR COUNTING
    Matrix d_error;
    d_error.width = 1; d_error.length = epochs;
    cudaMalloc(&d_error.data, epochs * sizeof(double));


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((y.width + dimBlock.x - 1) / dimBlock.x, (X.length + dimBlock.y - 1) / dimBlock.y);
    int blocksPerGrid = (y.width * y.length + threadsPerBlock - 1) / threadsPerBlock;

    for (int epoch = 0; epoch < epochs; epoch++) {

        //---------FORWARD PASS--------------------------
        //------------------------------------------------------------
        // Multyply X*w
        //------------------------------------------------------------
        MatMulKernel << < dimGrid, dimBlock >> > (d_X, d_w, d_pred);

        //------------------------------------------------------------
        // Add bias
        //------------------------------------------------------------
        AddMatrixRepKernel << < blocksPerGrid, threadsPerBlock >> > (d_pred, d_b);

        //------------------------------------------------------------
        // Calculate loss
        //------------------------------------------------------------
        SubMatrixKernel << < blocksPerGrid, threadsPerBlock >> > (d_y, d_pred, d_difference);


        //---------BACKWARD PASS--------------------------
        LossFunc << < 1, 1 >> > (d_error, d_difference, epoch);
        bias_update << < blocksPerGrid, threadsPerBlock >> > (d_difference, d_b, THETA);
        TransposeKernelRep << < dimGrid, dimBlock >> > (d_X);
        MatMulKernel << < dimGrid, dimBlock >> > (d_X, d_difference, d_wGrad);
        TransposeKernelRep << < dimGrid, dimBlock >> > (d_X);
        weights_update << < blocksPerGrid, threadsPerBlock >> > (d_w, d_wGrad, THETA);
    }


    Matrix result;
    result.width = y.width; result.length = y.length;
    result.data = new double[result.width * result.length];
    cudaMemcpy(result.data, d_pred.data, result.width * result.length * sizeof(double), cudaMemcpyDeviceToHost);


    losses.width = 1; losses.length = epochs;
    losses.data = new double[epochs];
    cudaMemcpy(losses.data, d_error.data, epochs * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_X.data);
    cudaFree(d_y.data);

    cudaFree(d_b.data);
    cudaFree(d_b.data);
    cudaFree(d_w.data);
    cudaFree(d_wGrad.data);
    cudaFree(d_difference.data);

    return result;
};