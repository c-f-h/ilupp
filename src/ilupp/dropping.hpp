#pragma once

#include "declarations.h"

namespace iluplusplus {

template<class T>
void threshold_and_drop(const vector_sparse_dynamic<T>& w, std::vector<Integer>& list, Integer n, T tau, Integer from, Integer to)
{
    list.clear();
    if (n <= 0)
        return;     // return empty list

    const Real norm = w.norm2(from, to);

    // get indices which are in [from,to) and above the threshold
    for (Integer x = 0; x < w.non_zeroes(); ++x) {
        const Integer i = w.get_pointer(x);
        if (from <= i && i < to && std::abs(w.get_data(x)) > norm * tau)
            list.push_back(x);
    }

    if (list.size() > n) {
        // too many candidates -- sort by decreasing absolute size
        std::sort(list.begin(), list.end(),
                [&](Integer x, Integer y) { return std::abs(w.get_data(x)) > std::abs(w.get_data(y)); });

        list.resize(n);
    }

    // now size() <= n - sort by increasing column index
    std::sort(list.begin(), list.end(),
            [&](Integer x, Integer y) { return std::abs(w.get_pointer(x)) < std::abs(w.get_pointer(y)); });
}

}
