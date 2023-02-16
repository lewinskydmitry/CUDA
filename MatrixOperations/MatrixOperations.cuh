#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <cmath>

#include "../Matrix/Matrix.h"

#define BLOCK_SIZE 16
#define threadsPerBlock 128

//-------------------------------------------------------------
// GPU MATRIX ADDITION
//-------------------------------------------------------------
__global__ void AddMatrixKernel(const Matrix A, const Matrix B, Matrix C);
Matrix AddMatrix(Matrix A, Matrix B);
__global__ void AddMatrixRepKernel(Matrix A, Matrix B);
void AddMatrixRep(Matrix& A, Matrix B);

//-------------------------------------------------------------
// GPU MATRIX SUBSTRACTION
//-------------------------------------------------------------
__global__ void SubMatrixKernel(const Matrix A, const Matrix B, Matrix C);
Matrix SubMatrix(Matrix A, Matrix B);
__global__ void SubMatrixRepKernel(Matrix& A, Matrix B);
void SubMatrixRep(Matrix& A, Matrix B);

//-------------------------------------------------------------
// GPU NAIVE MATRICES MULTIPLICATION
//-------------------------------------------------------------
__global__ void MatMulNaiveKernel(Matrix A, Matrix B, Matrix C);
Matrix MatMulNaive(Matrix A, Matrix B);

//-------------------------------------------------------------
// SHARED MEMORY GPU MATRICES MULTIPLICATION
//-------------------------------------------------------------
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C);
Matrix MatMul(const Matrix A, const Matrix B);

//-------------------------------------------------------------
// SHARED MEMORY GPU MATRIX TRANSPOSE
//-------------------------------------------------------------
__global__ void TransposeKernel(Matrix odata, const Matrix idata);
Matrix Transpose(Matrix A);
__global__ void TransposeKernelRep(Matrix Matrixdata);
void TransposeRep(Matrix& A);
