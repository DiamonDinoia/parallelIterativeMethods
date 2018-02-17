#include <iostream>
#include <Eigen>
#include "denseBlocksJacobi.h"
#include "denseOverlappingJacobi.h"
#include "asyncBlocksJacobi.h"
#include "asyncOverlappingJacobi.h"
#include "asyncJacobi.h"
#include "jacobi.h"
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
//	ColumnVector<float, Dynamic> test2(5);
//
//	jacobi<float, 4> marco(matrix, test, 100, 0.f, 8);
//
//	auto tmp = marco.solve();

	const int size = 5;
	const int iterations = 100;
	const auto tolerance = 0.000001f;
	const auto workers = 8;
	const auto blockSize = 2;


//    Matrix<float, size, size> A = Matrix<float, size, size>::Random();
    Matrix<float, Dynamic, Dynamic> A = Matrix<float, Dynamic, Dynamic>::Random(size,size);

	ColumnVector<float, Dynamic> b = ColumnVector<float, Dynamic>::Random(size);

    for (int i = 0; i < A.rows(); ++i) {
        A.row(i)=A.row(i)/A.row(i).maxCoeff();
        float value = A.row(i).template lpNorm<1>();
        value-=A(i,i);
        A(i,i) = value;

    }
	jacobi<float, Dynamic> jacobi(A, b, iterations, tolerance, workers);
    asyncJacobi<float, Dynamic> asyncJacobi(A, b, iterations, tolerance, workers);
	denseBlocksJacobi<float , Dynamic> denseBlocksJacobi(A, b, iterations, tolerance, workers, blockSize);
	denseOverlappingJacobi<float , Dynamic> denseOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);
    asyncBlocksJacobi<float, Dynamic> asyncBlocksJacobi(A, b, iterations, tolerance, workers, blockSize, 0);
    asyncOverlappingJacobi<float , Dynamic> asyncOverlappingJacobi(A, b, iterations, tolerance, workers, blockSize);

    auto error = 0.f;

    auto static start_time = Time::now();
    error = (b-A*jacobi.solve()).template lpNorm<1>()/size;
    auto static end_time = Time::now();
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

//    cout << jacobi.solve().transpose() << endl;
//
//    cout << denseBlocksJacobi.solve().transpose() << endl;
//
//    cout << denseOverlappingJacobi.solve().transpose() << endl;
//
//    cout << asyncBlocksJacobi.solve().transpose() << endl;
//
//    cout << asyncOverlappingJacobi.solve().transpose() << endl;
////
	

//    generate_diagonal_dominant_matrix<Matrix<float, 5, 5>>(ref(random));

//    std::cout << A << endl;


//    for (int i = 0; i < 10; ++i) {
//
//    denseBlocksJacobi<float , 4 > marco2(matrix, test, 100, 0.000001, 8, 2);
//    denseOverlappingJacobi<float , 4 > marco3(matrix, test, 100, 0.000001, 8, 2);
//    asyncBlocksJacobi<float, 4> marco4(matrix, test, 100, 0.000001, 8, 2, 0);
//    asyncOverlappingJacobi<float , 4 > marco5(matrix, test, 100, 0.000001, 8, 2);
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
