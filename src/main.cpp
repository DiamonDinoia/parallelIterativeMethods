#include <iostream>
#include <Eigen>
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
//#include "test.h"


using namespace std;
using namespace Eigen;
using namespace Iterative;

//template<T> SquareMatrix<float>{};

int main(const int argc, const char* argv[]) {

    Eigen::setNbThreads(1);

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

	const int size = 1024;
	const int iterations = 100;
	const auto tolerance = 0.0000001f;
	const auto workers = 8;
    auto blockSize = 3;


    auto error = 0.;

	ColumnVector<double, Dynamic> b = ColumnVector<double, Dynamic>::Random(size);

    #ifdef Dense

//    Matrix<float, size, size> A = Matrix<float, size, size>::Random();
    Matrix<double, Dynamic, Dynamic> A =
            Matrix<double, Dynamic, Dynamic>::Random(size,size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A.rows(); ++i) {
//        A.row(i)*=10000000;
//        A(i,i)=0.0000000005;
        auto value = A.row(i).template lpNorm<1>();
//        value-=A(i,i);
        A(i,i) = value;

    }
	denseSerialJacobi<double, Dynamic> serialJacobi(A, b, iterations, tolerance);
	denseParallelJacobi<double, Dynamic> jacobi(A, b, iterations, tolerance, workers);
    denseAsyncJacobi<double, Dynamic> asyncJacobi(A, b, iterations, tolerance, workers);
    denseBlocksJacobi<double , Dynamic> denseBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseOverlappingJacobi<double , Dynamic> denseOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseAsyncBlocksJacobi<double, Dynamic> asyncBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    denseAsyncOverlappingJacobi<double , Dynamic> asyncOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);    Matrix<double, Dynamic, Dynamic> A =


    #endif


    #ifdef SPARSE



    SparseMatrix<double>A(size,size);

    typedef Eigen::Triplet<double> T;

    vector<T> triplets;

    for (int i = 0; i < 4*size; ++i) {
        triplets.emplace_back(T(abs(rand())%size,abs(rand())%size,(double)rand()/RAND_MAX));
        if(i<size)
            triplets.emplace_back(T(i,i,(double)rand()/RAND_MAX));
    }

    A.setFromTriplets(triplets.begin(),triplets.end());


//    for (int i = 0; i < size*10; ++i) {
//        A.insert(Eigen::Index(rand()%size),Eigen::Index(rand()%size))=(double)rand();
//    }



//    for (int i = 0; i < size; ++i) {
//        b.insert(i) = (double)rand();
//    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; ++i) {
//        A.row(i)*=10000000;
//        A(i,i)=0.0000000005;
        auto sum=0.;
        for (int j = 0; j < size; ++j) {
            sum+=abs(A.coeff(i,j));
        }
        //        value-=A(i,i);
        A.coeffRef(i,i) = sum;


    }

//    A.makeCompressed();

	sparseSerialJacobi<double> denseSerialJacobi(A, b, iterations, tolerance);
	sparseParallelJacobi<double> jacobi(A, b, iterations, tolerance, workers);
    sparseAsyncJacobi<double> asyncJacobi(A, b, iterations, tolerance, workers);
    sparseBlocksJacobi<double> denseBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    sparseOverlappingJacobi<double> denseOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
    sparseAsyncBlocksJacobi<double> asyncBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
    sparseAsyncOverlappingJacobi<double> asyncOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);

    #endif

    auto start_time = Time::now();
    error = (b-A*denseSerialJacobi.solve()).template lpNorm<1>()/size;
    auto end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    start_time = Time::now();
    error = (b-A*jacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    start_time = Time::now();
    error = (b-A*asyncJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    start_time = Time::now();
    error = (b-A*denseBlocksJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    start_time = Time::now();
    error = (b-A*denseOverlappingJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

    start_time = Time::now();
    error = (b-A*asyncBlocksJacobi.solve()).template lpNorm<1>()/size;
    end_time = Time::now();
    cout << "error: " << error << endl;
    std::cout << "time: " << ' ' << dsec(end_time - start_time).count() << std::endl;

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
