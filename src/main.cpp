#include <iostream>
#include <Eigen>
#include <fstream>
#include <unistd.h>
#include <cstdlib>
#include "denseBlocksJacobi.h"
#include "denseOverlappingJacobi.h"
#include "denseAsyncBlocksJacobi.h"
#include "denseAsyncOverlappingJacobi.h"
#include "denseAsyncJacobi.h"
#include "denseParallelJacobi.h"
#include "denseSerialJacobi.h"
#include "sparseBlocksJacobi.h"
#include "sparseOverlappingJacobi.h"
#include "sparseAsyncBlocksJacobi.h"
#include "sparseAsyncOverlappingJacobi.h"
#include "sparseAsyncJacobi.h"
#include "sparseParallelJacobi.h"
#include "sparseSerialJacobi.h"
#include "parser.h"
#include "sparseOptimizedBlocksJacobi.h"
#include "sparseOptimizedOverlappingJacobi.h"
#include "denseOptimizedBlocksJacobi.h"
#include "denseOptimizedOverlappingJacobi.h"

using namespace std;
using namespace Eigen;
using namespace Iterative;

//template<T> SquareMatrix<float>{};

auto debug = false;
auto toCsv = false;
string filename;

ulong matrixSize = 1024;
ulong iterations = 100;
double tolerance = 0.000000001;
//	const auto tolerance = 0.0;
//	const auto tolerance = 0.00000000000000000001;
int workers = 8;
auto blockSize = 64;

const auto sequential = "sequential";
const auto parallel = "parallel";
const auto parallel_async = "parallel_async";
const auto blocks = "blocks";
const auto blocks_optimized = "blocks_optimized";
const auto blocks_async = "blocks_async";
const auto overlapping = "overlapping";
const auto overlapping_optimized = "overlapping_optimized";
const auto overlapping_async = "overlapping_async";


enum methods {
    SEQUENTIAL,
    PARALLEL,
    PARALLEL_ASYNC,
    BLOCKS,
    BLOCKS_OPTIMIZED,
    BLOCKS_ASYNC,
    OVERLAPPING,
    OVERLAPPING_OPTIMIZED,
    OVERLAPPING_ASYNC
};

auto method = SEQUENTIAL;
string methodString = sequential;

void parse_args(int argc, char *argv[]);
void write_csv(string fileName, std::chrono::duration<double> time, double error, long iteration);

int main(int argc, char *argv[]) {

    parse_args(argc, argv);

    Eigen::setNbThreads(workers);
    auto error = 0.;

    chrono::time_point<chrono::high_resolution_clock> start_time;
    chrono::time_point<chrono::high_resolution_clock> end_time;
    ColumnVector<double, Dynamic> x;
    srand(42);

#ifdef DENSE


    Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Random(matrixSize, matrixSize);
    ColumnVector<double, Dynamic> b = ColumnVector<double, Dynamic>::Random(matrixSize);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < A.rows(); ++i) {
        auto value = A.row(i).template lpNorm<1>();
        A(i, i) = value;
    }

    denseSerialJacobi<double, Dynamic> serialJacobi(A, b, iterations, tolerance);
    denseParallelJacobi<double, Dynamic> parallelJacobi(A, b, iterations, tolerance, workers);
    denseAsyncJacobi<double, Dynamic> asyncJacobi(A, b, iterations, tolerance, workers);
    denseBlocksJacobi<double, Dynamic> blocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseOptimizedBlocksJacobi<double, Dynamic> optimizedBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseOverlappingJacobi<double, Dynamic> overlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseOptimizedOverlappingJacobi<double, Dynamic> optimizedOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseAsyncBlocksJacobi<double, Dynamic> asyncBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseAsyncOverlappingJacobi<double, Dynamic> asyncOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);


#endif


#ifdef SPARSE


    //    SparseMatrix<double>A;
    //    ifstream input(argv[1]);
    //    read_matrix<double>(A, input);
    //    input.close();

        SparseMatrix<double>A(matrixSize,matrixSize);

        ColumnVector<double, Dynamic> b = ColumnVector<double, Dynamic>::Zero(A.cols());

        typedef Eigen::Triplet<double> T;

        vector<T> triplets;

        for (int i = 0; i < 100*matrixSize; ++i) {
            triplets.emplace_back(T(abs(rand())%matrixSize,abs(rand())%matrixSize,(double)rand()*1000/RAND_MAX));
            if(i<matrixSize)
                triplets.emplace_back(T(i,i,(double)rand()*1000/RAND_MAX));
        }

        A.setFromTriplets(triplets.begin(),triplets.end());

        triplets.clear();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < matrixSize; ++i) {
            auto sum=0.;
            for (int j = 0; j < matrixSize; ++j) {
                sum+=abs(A.coeff(i,j));
            }
            A.coeffRef(i,i) = sum;
        }

    //    A.makeCompressed();

        sparseSerialJacobi<double> serialJacobi(A, b, iterations, tolerance);
        sparseParallelJacobi<double> parallelJacobi(A, b, iterations, tolerance, workers);
        sparseAsyncJacobi<double> asyncJacobi(A, b, iterations, tolerance, workers);
        sparseBlocksJacobi<double> blocksJacobi(A, b, iterations, tolerance, workers, blockSize);
        sparseOptimizedBlocksJacobi<double> optimizedBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
        sparseOverlappingJacobi<double> overlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
        sparseOptimizedOverlappingJacobi<double> optimizedOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
        sparseAsyncBlocksJacobi<double> asyncBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
        sparseAsyncOverlappingJacobi<double> asyncOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);

#endif

    auto iterationsPerformed = 0L;

    switch (method){

        case SEQUENTIAL:
            methodString = sequential;
            cout << "Sequential" << endl;
            start_time = Time::now();
            x = serialJacobi.solve();
            end_time = Time::now();
            error = (b-A*x).template lpNorm<1>()/matrixSize;
            cout << "error: " << error << endl;
            std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;
            iterationsPerformed = serialJacobi.getIteration();
            break;
        case PARALLEL:
            methodString = parallel;
            cout << "Parallel" << endl;
            start_time = Time::now();
            x = parallelJacobi.solve();
            end_time = Time::now();
            error = (b - A * x).template lpNorm<1>() / matrixSize;
            cout << "error: " << error << endl;
            std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;
            iterationsPerformed = parallelJacobi.getIteration();
            break;
        case PARALLEL_ASYNC:
            methodString = parallel_async;
            cout << "Parallel async" << endl;
            start_time = Time::now();
            x = asyncJacobi.solve();
            end_time = Time::now();
            error = (b - A * x).template lpNorm<1>() / matrixSize;
            cout << "error: " << error << endl;
            std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;
            iterationsPerformed = asyncJacobi.getIteration();
            break;
        case BLOCKS:
            methodString = blocks;
            cout << "Parallel blocks" << endl;
            start_time = Time::now();
            x = blocksJacobi.solve();
            end_time = Time::now();
            error = (b - A * x).template lpNorm<1>() / matrixSize;
            cout << "error: " << error << endl;
            std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;
            iterationsPerformed = blocksJacobi.getIteration();

            break;
        case BLOCKS_OPTIMIZED:
            methodString = blocks_optimized;
            cout << "Parallel Optimized blocks" << endl;
            start_time = Time::now();
            x = optimizedBlocksJacobi.solve();
            end_time = Time::now();
            error = (b - A * x).template lpNorm<1>() / matrixSize;
            cout << "error: " << error << endl;
            std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;
            iterationsPerformed = optimizedBlocksJacobi.getIteration();
            break;
        case BLOCKS_ASYNC:
            methodString = blocks_async;
            cout << "Parallel async blocks" << endl;
            start_time = Time::now();
            x = asyncBlocksJacobi.solve();
            end_time = Time::now();
            error = (b - A * x).template lpNorm<1>() / matrixSize;
            cout << "error: " << error << endl;
            std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;
            iterationsPerformed = asyncBlocksJacobi.getIteration();

            break;
        case OVERLAPPING:
            methodString = overlapping;
            cout << "Parallel overlapping" << endl;
            start_time = Time::now();
            x = overlappingJacobi.solve();
            end_time = Time::now();
            error = (b - A * x).template lpNorm<1>() / matrixSize;
            cout << "error: " << error << endl;
            std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;
            iterationsPerformed = overlappingJacobi.getIteration();
            break;
        case OVERLAPPING_OPTIMIZED:
            methodString = overlapping_optimized;
            cout << "Parallel optimized overlapping" << endl;
            start_time = Time::now();
            x = optimizedOverlappingJacobi.solve();
            end_time = Time::now();
            error = (b - A * x).template lpNorm<1>() / matrixSize;
            cout << "error: " << error << endl;
            std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;
            iterationsPerformed = optimizedOverlappingJacobi.getIteration();
            break;
        case OVERLAPPING_ASYNC:
            methodString = overlapping_async;
            cout << "Parallel async overlapping blocks" << endl;
            start_time = Time::now();
            x = asyncOverlappingJacobi.solve();
            end_time = Time::now();
            error = (b - A * x).template lpNorm<1>() / matrixSize;
            cout << "error: " << error << endl;
            std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;
            iterationsPerformed = asyncOverlappingJacobi.getIteration();
            break;
    }

    if(toCsv) write_csv(filename, dsec(end_time - start_time), error, iterationsPerformed);

    if(debug) cout << x.transpose() << endl;


//    cerr << "OK" << endl;

    return 0;

}

void parse_args(int argc,  char *argv[]) {

    string arg(argv[1]);
    if (arg == sequential) method = SEQUENTIAL;
    else if (arg == parallel) method = PARALLEL;
    else if (arg == parallel_async) method = PARALLEL_ASYNC;
    else if (arg == blocks) method = BLOCKS;
    else if (arg == blocks_optimized) method = BLOCKS_OPTIMIZED;
    else if (arg == blocks_async) method = BLOCKS_ASYNC;
    else if (arg == overlapping) method = OVERLAPPING;
    else if (arg == overlapping_optimized) method = OVERLAPPING_OPTIMIZED;
    else if (arg == overlapping_async) method = OVERLAPPING_ASYNC;
    else exit(1);

    InputParser input(argc,argv);

    if(input.cmdOptionExists("-w")){

        workers = (int) strtol(input.getCmdOption("-w").c_str(), nullptr, 10);
    }

    if(input.cmdOptionExists("-s")){

        matrixSize = (ulong) strtol(input.getCmdOption("-s").c_str(), nullptr, 10);
    }

    if(input.cmdOptionExists("-i")){

        iterations = (ulong) strtol(input.getCmdOption("-i").c_str(), nullptr, 10);
    }

    if(input.cmdOptionExists("-t")){

        tolerance = stod(input.getCmdOption("-t"));
    }

    if(input.cmdOptionExists("-b")){

        blockSize = (int) strtol(input.getCmdOption("-b").c_str(), nullptr, 10);
    }

    if(input.cmdOptionExists("-p")){
        filename = input.getCmdOption("-p");
        toCsv = true;
        cout << filename << endl;
    }

    if(input.cmdOptionExists("-d")){
        debug = true;
    }

}

void write_csv(string fileName, std::chrono::duration<double> time, double error, long iteration){

    ifstream test(fileName);
    auto exists = test.good();
    test.close();
    ofstream outFile(fileName, ofstream::out | ofstream::app);

    if(!exists){
        outFile << "algorithm,time,iterations,workers,error,size" << endl;
    }

    outFile << methodString << ',' << time.count() << ',' << iteration <<','<< workers << ',' << error << ','
            << matrixSize << endl;

}
