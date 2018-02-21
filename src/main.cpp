#include <iostream>
#include <Eigen>
#include <fstream>
#include "denseBlocksJacobi.h"
#include "denseFixedBlocksJacobi.h"
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
#include "sparseFixedBlocksJacobi.h"
//#include "test.h"
#include "parser.h"

using namespace std;
using namespace Eigen;
using namespace Iterative;

//template<T> SquareMatrix<float>{};

int main(const int argc, const char* argv[]) {

    Eigen::setNbThreads(1);


    ifstream input(argv[1]);

	//std::cout << "Hello, World!" << std::endl;

	//Matrix<float, 5, 5> A;

//	Matrix<float, 4, 4> matrix;
//	matrix << 10., -1., 2., 0.,
//			  -1., 11., -1., 3.,
//			  2., -1., 10., -1.,
//			  0., 3., -1., 8.;
//	Matrix<float, Dynamic, Dynamic> matrix1(3, 3);
//
//	ColumnVector<float, 4> test;
//
//	test << 6., 25., -11., 15.;
//
//    ColumnVector<float, Dynamic> test2(5);
//
//	denseParallelJacobi<float, 4> marco(matrix, test, 100, 0.f, 8);
//
//	auto tmp = marco.solve();

	const int size = 16384;
	const int iterations = 5;
	const auto tolerance = 0.000000;
	const auto workers = 4;
    const auto blockSize = 256;


    auto error = 0.;


    #ifdef DENSE

    Matrix<double, Dynamic, Dynamic> A = Matrix<double, Dynamic, Dynamic>::Random(size,size);
    ColumnVector<double, Dynamic> b = ColumnVector<double, Dynamic>::Zero(size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A.rows(); ++i) {
        auto value = A.row(i).template lpNorm<1>();
        A(i,i) = value;
    }

//	denseSerialJacobi<double, Dynamic> serialJacobi(A, b, iterations, tolerance);
//	denseParallelJacobi<double, Dynamic> parallelJacobi(A, b, iterations, tolerance, workers);
//    denseAsyncJacobi<double, Dynamic> asyncJacobi(A, b, iterations, tolerance, workers);
    denseBlocksJacobi<double , Dynamic> blocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseFixedBlocksJacobi<double , Dynamic, blockSize> fixedBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseOverlappingJacobi<double , Dynamic> overlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseAsyncBlocksJacobi<double, Dynamic> asyncBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseAsyncOverlappingJacobi<double , Dynamic> asyncOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);


    #endif


    #ifdef SPARSE

    srand(42);

//    SparseMatrix<double>A;
//
//    read_matrix<double>(A, input);
//
//    ColumnVector<double, Dynamic> b = ColumnVector<double, Dynamic>::Zero(A.cols());

//    A = A.transpose();

    SparseMatrix<double>A(size,size);

//    read_matrix<double>(A, input);

    ColumnVector<double, Dynamic> b = ColumnVector<double, Dynamic>::Zero(A.cols());

    typedef Eigen::Triplet<double> T;

    vector<T> triplets;

    for (int i = 0; i < 100*size; ++i) {
        triplets.emplace_back(T(abs(rand())%size,abs(rand())%size,(double)rand()*1000/RAND_MAX));
        if(i<size)
            triplets.emplace_back(T(i,i,(double)rand()*1000/RAND_MAX));
    }

    A.setFromTriplets(triplets.begin(),triplets.end());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
        auto sum=0.;
        for (int j = 0; j < size; ++j) {
            sum+=abs(A.coeff(i,j));
        }
        A.coeffRef(i,i) = sum;
    }

    A.makeCompressed();

//	sparseSerialJacobi<double> serialJacobi(A, b, iterations, tolerance);
//	sparseParallelJacobi<double> parallelJacobi(A, b, iterations, tolerance, workers);
//    sparseAsyncJacobi<double> asyncJacobi(A, b, iterations, tolerance, workers);
    sparseBlocksJacobi<double> blocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    sparseFixedBlocksJacobi<double, blockSize> fixedBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    sparseOverlappingJacobi<double> overlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
    sparseAsyncBlocksJacobi<double> asyncBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    sparseAsyncOverlappingJacobi<double> asyncOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);

    #endif

    cout << "Sequential" << endl;
    auto start_time = Time::now();
//    error = (b-A*serialJacobi.solve()).template lpNorm<1>()/size;
    auto end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

//    cout << serialJacobi.getSolution().transpose() << endl;

    cout << "Parallel" << endl;
    start_time = Time::now();
//    error = (b-A*parallelJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel async" << endl;
    start_time = Time::now();
//    error = (b-A*asyncJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel blocks" << endl;
    start_time = Time::now();
    error = (b-A*blocksJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel fixed blocks" << endl;
    start_time = Time::now();
    error = (b-A*fixedBlocksJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel overlapping" << endl;
    start_time = Time::now();
    error = (b-A*overlappingJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel async blocks" << endl;
    start_time = Time::now();
    error = (b-A*asyncBlocksJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    cout << "Parallel async overlapping blocks" << endl;
    start_time = Time::now();
    error = (b-A*asyncOverlappingJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;


//    cout << denseParallelJacobi.solve().transpose() << endl;
//
//    cout << denseBlocksJacobi.solve().transpose() << endl;
//
//    cout << denseOverlappingJacobi.solve().transpose() << endl;
//
//    cout << denseAsyncBlocksJacobi.solve().transpose() << endl;
//
//    cout << denseAsyncOverlappingJacobi.solve().transpose() << endl;

//    generate_diagonal_dominant_matrix<Matrix<float, 5, 5>>(ref(random));

//    std::cout << A << endl;


//    for (int i = 0; i < 10; ++i) {
//
//    denseBlocksJacobi<float , 4 > marco2(matrix, test, 100, 0.000001, 8, 2);
//    denseOverlappingJacobi<float , 4 > marco3(matrix, test, 100, 0.000001, 8, 2);
//    denseAsyncBlocksJacobi<float, 4> marco4(matrix, test, 100, 0.000001, 8, 2, 0);
//    denseAsyncOverlappingJacobi<float , 4 > marco5(matrix, test, 100, 0.000001, 8, 2);
//
//    cout << marco2.solve().transpose() << endl;
//
//    cout << marco3.solve().transpose() << endl;
//
//    cout << marco4.solve().transpose() << endl;
//
//    cout << marco5.solve().transpose() << endl;//    
//
//    }

//    create_matrix<SparseMatrix, float, 4>;

//	int ok;
//	cin >> ok;
	return 0;

}
