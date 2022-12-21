#pragma once

#include <vector>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>


class DataLoader {
public:
    size_t length;
    size_t width;
    std::vector< std::vector<double> > vector2d;
    double* data;

    DataLoader(std::string, char);
    void print();
    void create_data();
    static void print_matrix(double* matrix, int length, int width);
};