//
// Created by mbarb on 19/02/2018.
//

#ifndef PARALLELITERATIVE_PARSER_H
#define PARALLELITERATIVE_PARSER_H

#include "parser.h"
#include <regex>
#include <string>
#include "utils.h"
#include <omp.h>

template <typename T>
void read_matrix(Eigen::SparseMatrix<T>& matrix, std::ifstream& file){
    long nonZeros;
    {
        long rows, cols;

        file >> rows >> cols >> nonZeros;
//        matrix.resize(rows, cols);
        matrix.resize(cols, rows);
        matrix.resizeNonZeros(nonZeros);
        std::string line;
        std::getline(file, line);
    }

    std::cout << nonZeros << std::endl;

    std::string row, col, value;
    for (auto i = 0; i < nonZeros; ++i) {
        file >> row >> col >> value;
        matrix.insert(strtol(col.c_str(),NULL,10)-1,strtol(row.c_str(),NULL,10)-1) = (T)stod(value);
//        matrix.insert(strtol(col.c_str(),NULL,10)-1,strtol(row.c_str(),NULL,10)) = (T)stod(value);
        row.clear();
        col.clear();
        value.clear();
    }

    std::cout << "parsed" << std::endl;

//    matrix.makeCompressed();
//
//    std::cout << "compressed" << std::endl;


}

#endif //PARALLELITERATIVE_PARSER_H
