#pragma once

#include <vector>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>


class Matrix {
public:
    size_t length;
    size_t width;
    size_t stride;
    double* data;

    static Matrix read_csv(std::string path, char delimiter);
    void print();
    void create_data(std::vector< std::vector<double> > vector2d, Matrix matrix);
    static void print_matrix(Matrix matrix);
    static Matrix create_matrix(int length, int width, int min, int max);
    int equal(Matrix comp_matrix);
    void T();
    static Matrix transpose(Matrix init);
    static int equal(Matrix A, Matrix B);
};