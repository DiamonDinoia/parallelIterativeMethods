//
// Created by mbarb on 19/02/2018.
//

//#include "parser.h"
//#include <regex>
//#include "utils.h"
//
//auto const endFile = "  -1  -1  -1";
//
//using namespace std;
//
//template <typename T>
//void read_matrix(Eigen::SparseMatrix<T>& matrix, ifstream& file){
//
//    const regex real("((\\+|-)?[[:digit:]]+)(\\.(([[:digit:]]+)?))?");
//    const regex integer("(\\+|-)?[[:digit:]]+");
//
//    string line;
//
//    while(file >> line){
//        if (line==endFile) break;
//
//        smatch intMatches;
//        smatch realMatches;
//
//        match_results(line, intMatches, integer);
//        match_results(line, realMatches, real);
//
//        if (intMatches.matrixSize()==3){
//            matrix.resize(stol(intMatches[0]), stol(intMatches[1]));
//            matrix.resizeNonZeros(stol(intMatches[2]));
//        }
//
//        if (intMatches.matrixSize()==2 && realMatches.matrixSize()==1){
//            matrix.insert(stol(intMatches[0]), stol(intMatches[1]), stod(realMatches[0]));
//        }
//
//
//        line.clear();
//    }
//
//
//
//}