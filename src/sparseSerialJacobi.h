//
// Created by mbarb on 17/02/2018.
//

#ifndef PARALLELITERATIVE_SPARSESERIALJACOBI_H
#define PARALLELITERATIVE_SPARSESERIALJACOBI_H

#include <Eigen>
#include <iostream>
#include "utils.h"

namespace Iterative {

    template <typename Scalar>
    class sparseSerialJacobi {

    public:
        /**
         *
         * @param A linear system matrix of max rank
         * @param b known terms vector
         * @param iterations  max number of iterations
         * @param tolerance min error tolerated
         */
        explicit sparseSerialJacobi(
                const Eigen::SparseMatrix<Scalar>& A,
                const Eigen::ColumnVector<Scalar, Eigen::Dynamic>& b,
                const ulonglong iterations,
                const Scalar tolerance) :
                A(A), b(b), iterations(iterations), tolerance(tolerance), solution(b) {

            solution.fill((Scalar)1/solution.size());
        }

        Eigen::ColumnVector<Scalar, Eigen::Dynamic> solve() {


            Eigen::ColumnVector<Scalar, Eigen::Dynamic> oldSolution(solution);

            std::vector<ulonglong> index(solution.size());

            for (ulonglong i = 0; i < solution.size(); ++i)
                index[i]=i;

            std::vector<ulonglong> remove;

            auto iteration = 0L;

            for (iteration = 0; iteration < iterations; ++iteration) {
                // initialize the error
                Scalar error = tolerance - tolerance;
                //calculate solutions parallelizing on rows
                for (long long i = 0; i < index.size(); ++i){
                    auto el = index[i];
                    solution[el] = solution_find(b[el], el, oldSolution);
                    error = solution[el]-oldSolution[el];
                    if(error <= tolerance){
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

    protected:

        const Eigen::SparseMatrix<Scalar>& A;
        const Eigen::ColumnVector<Scalar, Eigen::Dynamic>& b;


        const ulonglong iterations;
        const Scalar tolerance;

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

    public:
        const Eigen::ColumnVector<Scalar, -1> &getSolution() const {
            return solution;
        }

    };
};
#endif //PARALLELITERATIVE_SERIALJACOBI_H
