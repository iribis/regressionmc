#pragma once
#include "multiarray-crtp.h"
#include "multiarray.h"
#include "../quadrature/multidimensional-range.h"

template<typename M1, typename M2>
auto operator+(const multiarray_const<M1>& m1, const multiarray_const<M2>& m2) { 
	static_assert(M1::dimensions == M2::dimensions, "Addition of multiarrays with different number of dimensions"); 
	static_assert(M1::size == M2::size, "Addition of multiarrays with different sizes"); 
    std::array<std::size_t,M1::dimensions> resolution; resolution.fill(M1::size);
    multiarray<decltype(std::declval<M1::value_type>()+std::declval<M2::value_type>()),M1::size,M1::dimensions> sol;
    for (auto i : multidimensional_range(resolution))
        sol[i] = m1[i] + m2[i];
    return sol;
}

template<typename M, typename Factor>
auto operator*(const multiarray_const<M>& m, const Factor& f) { 
    std::array<std::size_t,M::dimensions> resolution; resolution.fill(M::size);
    multiarray<decltype(std::declval<M::value_type>()*f),M::size,M::dimensions> sol;
    for (auto i : multidimensional_range(resolution))
        sol[i] = m[i]*f;
    return sol;
}



