//
// Created by mbarb on 24/01/2018.
//

#ifndef PARALLELITERATIVE_JACOBI_H
#define PARALLELITERATIVE_JACOBI_H

#include <omp.h>

namespace Iterative {

	template <typename Scalar, long long SIZE>
	class jacobi {

	public:
        /**
         *
         * @param matrix linear system matrix of max rank
         * @param termsVector known terms vector
         * @param iterations  max number of iterations
         * @param tolerance min error tolerated
         * @param workers  number of threads
         */
		explicit jacobi(
            const Eigen::Matrix<Scalar, SIZE, SIZE>& matrix,
			const Eigen::ColumnVector<Scalar, SIZE>& termsVector,
			const ulonglong iterations,
			const Scalar tolerance,
			const ulong workers=0L) :
			    matrix(matrix), vector(termsVector), iterations(iterations), tolerance(tolerance),
			        workers(workers), solution(termsVector) {

            solution.setZero();
            omp_set_num_threads(workers);
		}

		virtual Eigen::ColumnVector<Scalar, SIZE> solve() {

			Eigen::ColumnVector<Scalar, SIZE> old_solution(solution);

			for (ulonglong iteration = 0; iteration < iterations; ++iteration) {
				// initialize the error
                Scalar error = tolerance - tolerance;
				//calculate solutions parallelizing on rows
				#pragma omp parallel for schedule(static)
				for (long long i = 0; i < solution.size(); ++i)
                    solution[i] = solution_find(vector[i], i);
				//compute the error norm 1 weighted on the size of the matrix
				error += (solution - old_solution).template lpNorm<1>();
				// check the error
				error /= solution.size();
				if (error <= tolerance) break;
				std::swap(solution, old_solution);
			}
			return solution;
		}

	protected:

		const Eigen::Matrix<Scalar, SIZE, SIZE>& matrix;
		const Eigen::ColumnVector<Scalar, SIZE>& vector;


		const ulonglong iterations;
		const Scalar tolerance;
		const ulong workers;

		Eigen::ColumnVector<Scalar, SIZE> solution;

	private:
		/**
		* utility function implementing the jacobi method in order to find one solution
		* @param row coeffiient row
		* @param solutions vector solution
		* @param term right term vector
		* @param index index of the solution
		* @return solution component
		*/
		inline Scalar solution_find(Scalar term, const ulonglong index) {
			term -= matrix.row(index) * solution;
			return (term + matrix(index, index) * solution[index]) / matrix(index, index);
		}

	};
};


#endif //PARALLELITERATIVE_JACOBI_H
