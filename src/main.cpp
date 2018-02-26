#include <iostream>
#include <Eigen>
#include <fstream>
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

int main(const int argc, const char* argv[]) {

    Eigen::setNbThreads(8);


	const int size = 1024;
	const int iterations = 1000;
	const auto tolerance = 0.000000001;
//	const auto tolerance = 0.0;
//	const auto tolerance = 0.00000000000000000001;
	const auto workers = 8;
    const auto blockSize = 256;

    auto error = 0.;

    chrono::time_point<chrono::high_resolution_clock> start_time;
    chrono::time_point<chrono::high_resolution_clock> end_time;
    ColumnVector<double,Dynamic> x;
    srand(42);

    #ifdef DENSE


    Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Random(size,size);
    ColumnVector<double, Dynamic> b = ColumnVector<double, Dynamic>::Random(size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A.rows(); ++i) {
        auto value = A.row(i).template lpNorm<1>();
        A(i,i) = value;
    }

	denseSerialJacobi<double, Dynamic> serialJacobi(A, b, iterations, tolerance);
	denseParallelJacobi<double, Dynamic> parallelJacobi(A, b, iterations, tolerance, workers);
    denseAsyncJacobi<double, Dynamic> asyncJacobi(A, b, iterations, tolerance, workers);
    denseBlocksJacobi<double , Dynamic> blocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseOptimizedBlocksJacobi<double, Dynamic> optimizedBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseOverlappingJacobi<double , Dynamic> overlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseOptimizedOverlappingJacobi<double , Dynamic> optimizedOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseAsyncBlocksJacobi<double, Dynamic> asyncBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseAsyncOverlappingJacobi<double , Dynamic> asyncOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);


    #endif


    #ifdef SPARSE


//    SparseMatrix<double>A;
//    ifstream input(argv[1]);
//    read_matrix<double>(A, input);
//    input.close();

    SparseMatrix<double>A(size,size);

    ColumnVector<double, Dynamic> b = ColumnVector<double, Dynamic>::Zero(A.cols());

    typedef Eigen::Triplet<double> T;

    vector<T> triplets;

    for (int i = 0; i < 100*size; ++i) {
        triplets.emplace_back(T(abs(rand())%size,abs(rand())%size,(double)rand()*1000/RAND_MAX));
        if(i<size)
            triplets.emplace_back(T(i,i,(double)rand()*1000/RAND_MAX));
    }

    A.setFromTriplets(triplets.begin(),triplets.end());

    triplets.clear();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        auto sum=0.;
        for (int j = 0; j < size; ++j) {
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

//    cout << "Sequential" << endl;
//    start_time = Time::now();
//    x = serialJacobi.solve();
//    end_time = Time::now();
//    error = (b-A*x).template lpNorm<1>()/size;
//    cout << "error: " << error << endl;
//    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel" << endl;
    start_time = Time::now();
    x = parallelJacobi.solve();
    end_time = Time::now();
    error = (b-A*x).template lpNorm<1>()/size;
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel async" << endl;
    start_time = Time::now();
    x = asyncJacobi.solve();
    end_time = Time::now();
    error = (b-A*x).template lpNorm<1>()/size;
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel blocks" << endl;
    start_time = Time::now();
    x = blocksJacobi.solve();
    end_time = Time::now();
    error = (b-A*x).template lpNorm<1>()/size;
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel Optimized blocks" << endl;
    start_time = Time::now();
    x = optimizedBlocksJacobi.solve();
    end_time = Time::now();
    error = (b-A*x).template lpNorm<1>()/size;
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel async blocks" << endl;
    start_time = Time::now();
    x = asyncBlocksJacobi.solve();
    end_time = Time::now();
    error = (b-A*x).template lpNorm<1>()/size;
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;


    cout << "Parallel overlapping" << endl;
    start_time = Time::now();
    x = overlappingJacobi.solve();
    end_time = Time::now();
    error = (b-A*x).template lpNorm<1>()/size;
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel optimized overlapping" << endl;
    start_time = Time::now();
    x = optimizedOverlappingJacobi.solve();
    end_time = Time::now();
    error = (b-A*x).template lpNorm<1>()/size;
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;


    cout << "Parallel async overlapping blocks" << endl;
    start_time = Time::now();
    x = asyncOverlappingJacobi.solve();
    end_time = Time::now();
    error = (b-A*x).template lpNorm<1>()/size;
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;


	return 0;

}
