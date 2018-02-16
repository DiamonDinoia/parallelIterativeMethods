#include <iostream>
#include <Eigen>
#include "denseBlocksJacobi.h"
#include "denseOverlappingJacobi.h"
#include "asyncBlocksJacobi.h"
#include "asyncOverlappingJacobi.h"
//#include "test.h"

using namespace std;
using namespace Eigen;
using namespace Iterative;

//template<T> SquareMatrix<float>{};

int main(const int argc, const char* argv[]) {

    Eigen::setNbThreads(1);

	//std::cout << "Hello, World!" << std::endl;

	//Matrix<float, 5, 5> A;

	Matrix<float, 4, 4> matrix;
	matrix << 10., -1., 2., 0.,
			  -1., 11., -1., 3.,
			  2., -1., 10., -1.,
			  0., 3., -1., 8.;
	Matrix<float, Dynamic, Dynamic> matrix1(3, 3);

	ColumnVector<float, 4> test;

	test << 6., 25., -11., 15.;

	ColumnVector<float, Dynamic> test2(5);

	jacobi<float, 4> marco(matrix, test, 100, 0.f, 8);

	auto tmp = marco.solve();




//    for (int i = 0; i < 10; ++i) {

    denseBlocksJacobi<float , 4 > marco2(matrix, test, 100, 0.000001, 8, 2);
    denseOverlappingJacobi<float , 4 > marco3(matrix, test, 100, 0.000001, 8, 2);
    asyncBlocksJacobi<float, 4> marco4(matrix, test, 100, 0.000001, 8, 2, 0);
    asyncOverlappingJacobi<float , 4 > marco5(matrix, test, 100, 0.000001, 8, 2);

    cout << marco2.solve().transpose() << endl;

    cout << marco3.solve().transpose() << endl;

    cout << marco4.solve().transpose() << endl;

    cout << marco5.solve().transpose() << endl;

//    }

//    create_matrix<SparseMatrix, float, 4>;
//
//	int ok;
//	cin >> ok;
	return 0;

}
