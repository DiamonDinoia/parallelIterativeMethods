//
// Created by mbarb on 17/02/2018.
//

#ifndef PARALLELITERATIVE_DENSESERIALJACOBI_H
#define PARALLELITERATIVE_DENSESERIALJACOBI_H

#include <Eigen>
#include <iostream>
#include "utils.h"


namespace Iterative {

    template <typename Scalar, long long SIZE>
    class denseSerialJacobi {

    public:
        /**
         *
         * @param A linear system matrix of max rank
         * @param b known terms vector
         * @param iterations  max number of iterations
         * @param tolerance min error tolerated
         */
        explicit denseSerialJacobi(
                const Eigen::Matrix<Scalar, SIZE, SIZE>& A,
                const Eigen::ColumnVector<Scalar, SIZE>& b,
                const ulonglong iterations,
                const Scalar tolerance) :
                A(A), b(b), iterations(iterations), tolerance(tolerance), solution(b) {

            solution.fill((Scalar)1/solution.size());
        }

        const Eigen::ColumnVector<Scalar, SIZE> solve() {


            Eigen::ColumnVector<Scalar, SIZE> oldSolution(solution);

            std::vector<ulonglong> index(solution.size());

            for (ulonglong i = 0; i < solution.size(); ++i)
                index[i]=i;

            std::vector<ulonglong> remove;


            for (iteration = 0; iteration < iterations; ++iteration) {
                //calculate solutions parallelizing on rows
                for (long long i = 0; i < index.size(); ++i){
                    auto el = index[i];
                    solution[el] = solution_find(b[el], el, oldSolution);
                    Scalar error = std::abs(solution[el]-oldSolution[el]);
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
            return this->solution;
        }

        const Eigen::ColumnVector<Scalar, SIZE> &getSolution() const {
            return solution;
        }

        const long getIteration() const {
            return iteration;
        }

    protected:

        const Eigen::Matrix<Scalar, SIZE, SIZE>& A;
        const Eigen::ColumnVector<Scalar, SIZE>& b;


        const ulonglong iterations;
        const Scalar tolerance;

        Eigen::ColumnVector<Scalar, SIZE> solution;

        long iteration = 0L;
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
#endif //PARALLELITERATIVE_SERIALJACOBI_H
