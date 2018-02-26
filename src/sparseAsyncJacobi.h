//
// Created by mbarb on 17/02/2018.
//

#ifndef PARALLELITERATIVE_SPARSEASYNCJACOBI_H
#define PARALLELITERATIVE_SPARSEASYNCJACOBI_H

#include <omp.h>
#include <Eigen>
#include <iostream>
#include "utils.h"

namespace Iterative {

    template <typename Scalar>
    class sparseAsyncJacobi {

    public:
        /**
         *
         * @param A linear system matrix of max rank
         * @param b known terms vector
         * @param iterations  max number of iterations
         * @param tolerance min error tolerated
         * @param workers  number of threads
         */
        explicit sparseAsyncJacobi(
                const Eigen::SparseMatrix<Scalar>& A,
                const Eigen::ColumnVector<Scalar, Eigen::Dynamic>& b,
                const ulonglong iterations,
                const Scalar tolerance,
                const ulong workers=0L) :
                A(A), b(b), iterations(iterations), tolerance(tolerance),
                workers(workers),solution(b) {

            solution.fill((Scalar)1/solution.size());
            omp_set_num_threads(workers);

        }

        const Eigen::ColumnVector<Scalar, Eigen::Dynamic> solve() {


            std::vector<ulonglong> index(solution.size());

            for (ulonglong i = 0; i < solution.size(); ++i)
                index[i]=i;

            std::vector<ulonglong> remove;

            auto iteration = 0L;

            for (iteration = 0; iteration < iterations; ++iteration) {
                //calculate solutions parallelizing on rows
                #pragma omp parallel
                #pragma omp for schedule(static) nowait
                for (long long i = 0; i < index.size(); ++i){
                    auto el = index[i];
                    Scalar oldElement = solution[el];
                    solution[el] = solution_find(b[el], el);
                    Scalar error = std::abs(solution[el]-oldElement);
                    if(error <= tolerance){
                        #pragma omp critical
                        remove.emplace_back(i);
                    }
                }

                if(!remove.empty()){
                    #pragma omp barrier
                    std::sort(remove.rbegin(), remove.rend());
                    for (auto i : remove) {
                        index.erase(index.begin() + i);
                    }
                    remove.clear();
                    if (index.empty()) break;
                }

            }
            std::cout << iteration << std::endl;
            return this->solution;
        }

    protected:

        const Eigen::SparseMatrix<Scalar>& A;
        const Eigen::ColumnVector<Scalar, Eigen::Dynamic>& b;


        const ulonglong iterations;
        const Scalar tolerance;
        const ulong workers;

        Eigen::ColumnVector<Scalar, Eigen::Dynamic> solution;

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
            return (term + A.coeff(index, index) * solution[index]) / A.coeff(index, index);
        }

    };
};

#endif //PARALLELITERATIVE_ASYNCJACOBI_H
