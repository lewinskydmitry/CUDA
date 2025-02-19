﻿#include "LogisticRegression.cuh"

// Kernel for broadcasing some value to whole array
__global__ void broadcast(Matrix matrix, double value)
{
    int size = matrix.width * matrix.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < size) {
        matrix.data[thread_idx] = value;
    }
}

// Kernel for sigmoid function
__global__ void LogisticFunction(Matrix predict)
{
    int size = predict.width * predict.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < size) {
        predict.data[thread_idx] = 1/(1-std::exp(predict.data[thread_idx]));
    }
}

// Threshold function for prediction
__global__ void TresholdFunc(Matrix predict)
{
    int size = predict.width * predict.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < size) {
        if (predict.data[thread_idx] >= 0.5) {
            predict.data[thread_idx] = 1;
        }
        else {
            predict.data[thread_idx] = 0;
        }
    }
}

// Kernel for decreasing learning rate if it's too big for this data. It allows avoid increasing losses
__global__ void step_control(Matrix d_error, Matrix weights, Matrix b,  int epoch, double* THETA)
{
    int blocksPerGrid_w = (weights.width * weights.length + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_b = (b.width * b.length + threadsPerBlock - 1) / threadsPerBlock;

    if (d_error.data[epoch - 1] > 1.2 * d_error.data[epoch] && epoch != 0) {
        *THETA = *THETA / 10;
        broadcast << < blocksPerGrid_w, threadsPerBlock >> > (weights, 0);
        broadcast << < blocksPerGrid_b, threadsPerBlock >> > (b, 0);
    }
}

// Kernel for updating weights using gradients
__global__ void weights_update(Matrix weights, Matrix weightsGrad, double* THETA, int length_data, double REG_TERM)
{
    int size = weightsGrad.width * weightsGrad.length;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < size) {
        weights.data[thread_idx] += *THETA * (weightsGrad.data[thread_idx] / length_data + REG_TERM * weights.data[thread_idx]);
    }
}

// Kernel for updating bias
__global__ void bias_update(Matrix difference, Matrix bias, double* THETA)
{
    int size = difference.width * difference.length;
    
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;

    if (i < size) {
        sdata[tid] = *THETA * difference.data[i] / difference.length;
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int blocksPerGrid_b = (bias.width * bias.length + threadsPerBlock - 1) / threadsPerBlock;
        broadcast << < blocksPerGrid_b, threadsPerBlock >> > (bias, bias.data[0] - sdata[0]);
    }
}

// Kernel for calculating regularization term for loss function (ridge)
__global__ void CalcRegTerm(Matrix weights, double* reg_value, double REG_TERM){

    int size = weights.width * weights.length;
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;

    if (i < size) {
        sdata[tid] =  pow(weights.data[i], 2);
    }

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *reg_value = REG_TERM * sdata[0];
    }
}

// Kernel for calculation loss for each epoch
__global__ void LossFuncRed(Matrix erMatrix, Matrix y_true, Matrix pred, int iteration, double* regvalue)  {

    int size = y_true.width * y_true.length;
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;

    if (i < size) {
        sdata[tid] = -1 * (y_true.data[i] * std::log(pred.data[i]) + (1 - y_true.data[i]) * (1 - std::log(pred.data[i]))) / pred.length;
    }
    
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        erMatrix.data[iteration] = sdata[0] + *regvalue;
    }
}


// The host function for fitting regression
Matrix LogisticRegression::fit(Matrix X, Matrix y, int epochs) {

    // Calculate transpose X matrix for future calculation
    Matrix XT = Transpose(X);
    // INIT MATRIXES AND VARIABLES
    Matrix d_X;
    d_X.width = X.width; d_X.length = X.length; d_X.stride = X.width;
    cudaMalloc(&d_X.data, X.width * X.length * sizeof(double));
    cudaMemcpy(d_X.data, X.data, X.width * X.length * sizeof(double), cudaMemcpyHostToDevice);

    double* THETA;
    cudaMalloc((void**)&THETA, sizeof(double));
    cudaMemcpy(THETA, &h_THETA, sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_XT;
    d_XT.width = XT.width; d_XT.length = XT.length; d_XT.stride = XT.width;
    cudaMalloc(&d_XT.data, XT.width * XT.length * sizeof(double));
    cudaMemcpy(d_XT.data, XT.data, XT.width * XT.length * sizeof(double), cudaMemcpyHostToDevice);

    Matrix d_y;
    d_y.width = y.width; d_y.length = y.length;  d_y.stride = y.width;
    cudaMalloc(&d_y.data, y.width * y.length * sizeof(double));
    cudaMemcpy(d_y.data, y.data, y.width * y.length * sizeof(double), cudaMemcpyHostToDevice);

    // WEIGHTS AND BIAS
    Matrix d_w;
    d_w.width = y.width; d_w.length = X.width; d_w.stride = y.width;
    cudaMalloc(&d_w.data, X.width * sizeof(double));
    cudaMemset(d_w.data, 0, X.width * sizeof(double));

    Matrix d_b;
    d_b.width = y.width; d_b.length = y.length; d_b.stride = y.width;
    cudaMalloc(&d_b.data, y.width * y.length * sizeof(double));

    // VARIABLES FOR OPTIMIZATION
    Matrix d_pred;
    d_pred.width = y.width; d_pred.length = y.length; d_pred.stride = y.width;
    cudaMalloc(&d_pred.data, y.width * y.length * sizeof(double));

    Matrix d_difference;
    d_difference.width = y.width; d_difference.length = y.length; d_difference.stride = y.width;
    cudaMalloc(&d_difference.data, y.width * y.length * sizeof(double));

    Matrix d_wGrad;
    d_wGrad.width = y.width; d_wGrad.length = X.width; d_wGrad.stride = y.width;
    cudaMalloc(&d_wGrad.data, X.width * sizeof(double));

    double* regvalue;
    cudaMalloc((void**)&regvalue, sizeof(double));
    cudaMemset(regvalue, 0, sizeof(double));

    // ERROR COUNTING
    Matrix d_error;
    d_error.width = 1; d_error.length = epochs;
    cudaMalloc(&d_error.data, epochs * sizeof(double));
    cudaMemset(&d_error, 0, epochs * sizeof(double));


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((y.width + dimBlock.x - 1) / dimBlock.x, (X.length + dimBlock.y - 1) / dimBlock.y);
    int blocksPerGrid = (y.width * y.length + threadsPerBlock - 1) / threadsPerBlock;

    
    for (int epoch = 0; epoch < epochs; epoch++) {

        //---------FORWARD PASS--------------------------
        // Calculate predict
        MatMulKernel << < dimGrid, dimBlock >> > (d_X, d_w, d_pred);
        // Add bias
        AddMatrixRepKernel << < blocksPerGrid, threadsPerBlock >> > (d_pred, d_b);
        LogisticFunction << < blocksPerGrid, threadsPerBlock >> > (d_pred);
        SubMatrixKernel << < blocksPerGrid, threadsPerBlock >> > (d_y, d_pred, d_difference);

        //---------BACKWARD PASS--------------------------
        // Calculate loss
        CalcRegTerm << < blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double) >> > (d_w, regvalue, REG_TERM);
        LossFuncRed << < blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double) >> > (d_error, d_y, d_pred, epoch, regvalue);
        // Update bias
        bias_update << < blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double) >> > (d_difference, d_b, THETA);
        // Update weights
        MatMulKernel << < dimGrid, dimBlock >> > (d_XT, d_difference, d_wGrad);
        weights_update << < blocksPerGrid, threadsPerBlock >> > (d_w, d_wGrad, THETA, y.length, REG_TERM);
        // check step
        step_control << < 1, 1 >> > (d_error, d_w, d_b, epoch, THETA);

        //----------------FINAL PREDICT------------------------
        TresholdFunc << < blocksPerGrid, threadsPerBlock >> > (d_pred);
    }
    

    // ASSIGNING DEVICE VARIABLES AFTER COMPUTING
    difference.width = y.width; difference.length = y.length;
    difference.data = new double[difference.width * difference.length];
    cudaMemcpy(difference.data, d_difference.data, difference.width * difference.length * sizeof(double), cudaMemcpyDeviceToHost);

    losses.width = 1; losses.length = epochs;
    losses.data = new double[epochs];
    cudaMemcpy(losses.data, d_error.data, epochs * sizeof(double), cudaMemcpyDeviceToHost);

    grads.width = y.width; grads.length = X.width;
    grads.data = new double[X.width];
    cudaMemcpy(grads.data, d_w.data, X.width * sizeof(double), cudaMemcpyDeviceToHost);

    b.width = y.width; b.length = y.length;
    b.data = new double[y.length];
    cudaMemcpy(b.data, d_b.data, y.length * sizeof(double), cudaMemcpyDeviceToHost);

    predict.width = y.width; predict.length = y.length;
    predict.data = new double[y.length];
    cudaMemcpy(predict.data, d_pred.data, y.length * sizeof(double), cudaMemcpyDeviceToHost);

    // FREE MEMORY
    cudaFree(d_X.data);
    cudaFree(d_XT.data);
    cudaFree(d_y.data);

    cudaFree(d_w.data);
    cudaFree(d_wGrad.data);
    cudaFree(d_b.data);
    cudaFree(d_pred.data);
    cudaFree(d_difference.data);
    cudaFree(d_error.data);
    cudaFree(THETA);

    return predict;
};