#pragma once

#include <vector>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>


class Matrix {
    // Memers of the matrix class
public:
    // length of the matrix
    size_t length;
    // Width of the matrix
    size_t width;
    // Stride is variable which we need for GPU matrix multiplication, it's should be equal width
    size_t stride;
    // Pointer to array data of the matrix
    double* data;
    
    // Function for reading csv files as matrix with length and width
    static Matrix read_csv(std::string path, char delimiter);
    // We need this function for transfer data from vector to array
    void create_data(std::vector< std::vector<double> > vector2d, Matrix matrix);
    // Function for printing matrix
    void print();
    // It's static function for printing matrix
    static void print_matrix(Matrix matrix);
    // static functon for creating random matrix
    static Matrix create_matrix(int length, int width, int min, int max);
    // Function for checking the similarity of matrices
    int equal(Matrix comp_matrix);
    // Static Function for checking the similarity of matrices
    static int equal(Matrix A, Matrix B);
    // Function for transpose matrix using CPU
    void T();
    // Static function for transpose matrix
    static Matrix transpose(Matrix init);
    // Static CPU matrix multiplication function
    static Matrix MatMul(Matrix A, Matrix B);
};