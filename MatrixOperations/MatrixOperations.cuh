#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include "../Matrix/Matrix.h"
#include <stdexcept>

#define BLOCK_SIZE 16
#define threadsPerBlock 256
#define THETA 0.00001

//-------------------------------------------------------------
// GPU MATRIX ADDITION
//-------------------------------------------------------------
void AddMatrixKernel(const Matrix A, const Matrix B, Matrix C);
Matrix AddMatrix(Matrix A, Matrix B);
void AddMatrixRepKernel(Matrix A, Matrix B);
Matrix AddMatrixRep(Matrix A, Matrix B);

//-------------------------------------------------------------
// GPU MATRIX SUBSTRACTION
//-------------------------------------------------------------
void SubMatrixKernel(const Matrix A, const Matrix B, Matrix C);
Matrix SubMatrix(Matrix A, Matrix B);
void SubMatrixRepKernel(Matrix A, Matrix B);
Matrix SubMatrixRep(Matrix A, Matrix B);

//-------------------------------------------------------------
// GPU MATRIX MULTIPLICATION
//-------------------------------------------------------------
__global__ void MatMulNaiveKernel(Matrix A, Matrix B, Matrix C);
Matrix MatMulNaive(Matrix A, Matrix B);

//-------------------------------------------------------------
// SHARED MEMORY GPU MATRIX MULTIPLICATION
//-------------------------------------------------------------
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C);
Matrix MatMul(const Matrix A, const Matrix B);

//-------------------------------------------------------------
// SHARED MEMORY GPU MATRIX TRANSPOSE
//-------------------------------------------------------------
__global__ void TransposeKernel(Matrix odata, const Matrix idata);
Matrix TransposeKernel(Matrix A);
