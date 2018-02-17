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


            Eigen::ColumnVector<Scalar, SIZE> oldSolution(solution);

            std::vector<ulonglong> index(solution.size());

            for (ulonglong i = 0; i < solution.size(); ++i)
                index[i]=i;

            std::vector<ulonglong> remove;

            auto iteration = 0L;

            for (iteration = 0; iteration < iterations; ++iteration) {
                // initialize the error
                Scalar error = tolerance - tolerance;
                //calculate solutions parallelizing on rows
                #pragma omp parallel for schedule(auto)
                for (long long i = 0; i < index.size(); ++i){
                    auto el = index[i];
                    solution[el] = solution_find(b[el], el, oldSolution);
                    error = solution[el]-oldSolution[el];
                    if(error <= tolerance){
                        #pragma omp critical
                        remove.emplace_back(i);
                    }
                }

                if(!remove.empty()){
                    std::sort(remove.rbegin(), remove.rend());
                    for (auto i : remove) {
                        index.erase(index.begin() + i);
                    }
                    remove.clear();
                    if (index.empty()) break;
                }

                //compute the error norm 1 weighted on the size of the A
//                error += (solution - oldSolution).template lpNorm<1>();
                // check the error
//                error /= solution.size();
//                if (error <= tolerance) break;
                std::swap(solution, oldSolution);
            }
            std::cout << iteration << std::endl;
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
        inline Scalar solution_find(Scalar term, const ulonglong index, Eigen::ColumnVector<Scalar,SIZE>& oldSolution) {
            term -= A.row(index) * oldSolution;
            return (term + A(index, index) * oldSolution[index]) / A(index, index);
        }

    };
};


#endif //PARALLELITERATIVE_JACOBI_H
