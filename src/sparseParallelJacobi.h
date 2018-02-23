//
// Created by mbarb on 24/01/2018.
//

#ifndef PARALLELITERATIVE_SPARSEPARALLELJACOBI_H
#define PARALLELITERATIVE_SPARSEPARALLELJACOBI_H

#include <omp.h>
#include <Eigen>
#include "utils.h"
#include "iostream"

namespace Iterative {

    template <typename Scalar>
    class sparseParallelJacobi {

    public:
        /**
         *
         * @param A linear system matrix of max rank
         * @param b known terms vector
         * @param iterations  max number of iterations
         * @param tolerance min error tolerated
         * @param workers  number of threads
         */
        explicit sparseParallelJacobi(
                const Eigen::SparseMatrix<Scalar>& A,
                const Eigen::ColumnVector<Scalar, Eigen::Dynamic>& b,
                const ulonglong iterations,
                const Scalar tolerance,
                const ulong workers=0L) :
                A(A), b(b), iterations(iterations), tolerance(tolerance),
                workers(workers), solution(b) {

            solution.fill((Scalar)1/solution.size());
            omp_set_num_threads(workers);
        }

        const Eigen::ColumnVector<Scalar, Eigen::Dynamic> solve() {


            Eigen::ColumnVector<Scalar, Eigen::Dynamic> oldSolution(solution);

            std::vector<ulonglong> index(solution.size());

            for (ulonglong i = 0; i < solution.size(); ++i)
                index[i]=i;

            std::vector<ulonglong> remove;

            auto iteration = 0L;

            for (iteration = 0; iteration < iterations; ++iteration) {
                //calculate solutions parallelizing on rows
                #pragma omp parallel for schedule(dynamic)
                for (long long i = 0; i < index.size(); ++i){
                    auto el = index[i];
                    solution[el] = solution_find(b[el], el, oldSolution);
                    Scalar error = std::abs(solution[el]-oldSolution[el]);
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

                std::swap(solution, oldSolution);
            }
            std::cout << iteration << std::endl;
            return solution;
        }

        const Eigen::ColumnVector<Scalar, -1> &getSolution() const {
            return solution;
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
        inline Scalar solution_find(Scalar term, const ulonglong index, Eigen::ColumnVector<Scalar, Eigen::Dynamic>& oldSolution) {
            term -= A.row(index) * oldSolution;
            return (term + A.coeff(index, index) * oldSolution[index]) / A.coeff(index, index);
        }

    };
};


#endif //PARALLELITERATIVE_JACOBI_H
