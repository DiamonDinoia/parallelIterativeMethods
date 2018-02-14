#include <iostream>
#include <Eigen>
#include "denseBlocksJacobi.h"
#include "denseOverlappingJacobi.h"

using namespace std;
using namespace Eigen;
using namespace Iterative;

//template<T> SquareMatrix<float>{};

int main(const int argc, const char* argv[]) {

	//std::cout << "Hello, World!" << std::endl;

	//Matrix<float, 5, 5> matrix;

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


    denseBlocksJacobi<float , 4 > marco2(matrix, test, 100, 0.f, 8, 2);

    denseOverlappingJacobi<float , 4 > marco3(matrix, test, 100, 0.f, 8, 2);

    cout << marco2.solve() << endl;

    cout << marco3.solve() << endl;

//	int ok;
//	cin >> ok;
	return 0;

}
