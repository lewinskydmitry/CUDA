#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

void cpu_matrix_mult(double* h_a, double* h_b, double* h_result, int m);