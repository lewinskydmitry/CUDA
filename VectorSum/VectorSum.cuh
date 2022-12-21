#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>


void cudaCallAddVectorKernel(const double* a,
    const double* b,
    double* c,
    const int size);