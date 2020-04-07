#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include "util.h"
#include "iterator_util.h"
#include "serialize.h"

std::string test(){
    std::cout << "Yai" << std::endl;
    return "Yippi";
}

std::vector<double> solveHarmonic(
    std::vector<double> & values,
    std::vector<double> const & derrivative_divisor,
    std::vector<bool> const & is_fixed,
    std::vector<size_t> & bounds,
    size_t max_iterations,
    double max_error)
{
    size_t size = std::accumulate(bounds.begin(), bounds.end(), 1, std::multiplies<size_t>());
    if (is_fixed.size() != size || derrivative_divisor.size() != size || values.size() != size)
    {
        throw std::runtime_error("Wrong input dimensions");
    }
    std::vector<double> values_copy = values;
    size_t dim = bounds.size();
    std::vector<size_t> step(1,1);
    step.reserve(dim);
    for (size_t i = 1; i < dim; ++i)
    {
        step[i] = step[i - 1] * bounds[i - 1];
    }
    std::vector<double> div_sums;
    div_sums.reserve(size);
    for (size_t index = 0; index < size; ++index)
    {
        double div_sum = 0;
        if (!is_fixed[index])
        {
            double derrivative_div_mid = derrivative_divisor[index];
            for (size_t i = 0, tmp = index; i < dim; ++i)
            {
                size_t position = tmp % bounds[i];
                tmp /= bounds[i];
                if (position > 0)
                {
                    size_t low = index - step[i];
                    assert(low < size);
                    double div = derrivative_div_mid - derrivative_divisor[low];
                    div_sum += (div = 1. / (1.+div * div));
                }
                if (position + 1 < bounds[i])
                {
                    size_t up = index + step[i];
                    assert(up < size);
                    double div = derrivative_div_mid - derrivative_divisor[up];
                    div_sum += (div = 1. / (1.+div * div));
                }
            }
        }
        div_sums.push_back(div_sum);
    }
    for (size_t iteration = 0; iteration < max_iterations; ++iteration)
    {
        double error = 0;
#pragma omp parallel
        {
            double local_error = 0;
    #pragma omp for
            for (size_t index = 0; index < size; ++index)
            {
                if (!is_fixed[index])
                {
                    double erg = 0;
                    double derrivative_div_mid = derrivative_divisor[index];
                    for (size_t i = 0, tmp = index; i < dim; ++i)
                    {
                        size_t position = tmp % bounds[i];
                        tmp /= bounds[i];
                        if (position > 0)
                        {
                            size_t low = index - step[i];
                            assert(low < size);
                            double div = derrivative_div_mid - derrivative_divisor[low];
                            erg += values[low] / (1.+div * div);;
                        }
                        if (position + 1 < bounds[i])
                        {
                            size_t up = index + step[i];
                            assert(up < size);
                            double div = derrivative_div_mid - derrivative_divisor[up];
                            erg += values[up] / (1.+div * div);
                        }
                    }
                    double div_sum = div_sums[index];
                    double add_middle = div_sum * values[index];
                    erg += add_middle;
                    erg /= div_sum * 2;
                    double difference = erg - add_middle;
                    local_error += difference * difference;
                    values_copy[index] = erg;
                }
            }
#pragma omp critical
    error += local_error;
        }
        values_copy.swap(values);
        if (error < max_error)
        {
            break;
        }
    }
    return values;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("solveHarmonic", &solveHarmonic, "A function which finds a Harmonic Function");
}

PYBIND11_MODULE(example2, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("test", &test, "A function which finds a Harmonic Function");
}
