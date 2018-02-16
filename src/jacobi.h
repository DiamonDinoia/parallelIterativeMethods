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
         * @param A linear system matrix of max rank
         * @param b known terms vector
         * @param iterations  max number of iterations
         * @param tolerance min error tolerated
         * @param workers  number of threads
         */
		explicit jacobi(
            const Eigen::Matrix<Scalar, SIZE, SIZE>& A,
			const Eigen::ColumnVector<Scalar, SIZE>& b,
			const ulonglong iterations,
			const Scalar tolerance,
			const ulong workers=0L) :
			    A(A), b(b), iterations(iterations), tolerance(tolerance),
			        workers(workers), solution(b) {

            solution.setZero();
            omp_set_num_threads(workers);
		}

		 Eigen::ColumnVector<Scalar, SIZE> solve() {

			Eigen::ColumnVector<Scalar, SIZE> old_solution(solution);

			for (ulonglong iteration = 0; iteration < iterations; ++iteration) {
				// initialize the error
                Scalar error = tolerance - tolerance;
				//calculate solutions parallelizing on rows
				#pragma omp parallel for schedule(static)
				for (long long i = 0; i < solution.size(); ++i)
                    solution[i] = solution_find(b[i], i);
				//compute the error norm 1 weighted on the size of the A
				error += (solution - old_solution).template lpNorm<1>();
				// check the error
				error /= solution.size();
				if (error <= tolerance) break;
				std::swap(solution, old_solution);
			}
			return solution;
		}

	protected:

		const Eigen::Matrix<Scalar, SIZE, SIZE>& A;
		const Eigen::ColumnVector<Scalar, SIZE>& b;


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
			term -= A.row(index) * solution;
			return (term + A(index, index) * solution[index]) / A(index, index);
		}

	};
};


#endif //PARALLELITERATIVE_JACOBI_H
