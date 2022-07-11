#pragma once

#include <vector>
#include <array>
#include "multiarray-crtp.h"
#include "../quadrature/multidimensional-range.h"


template<typename T, std::size_t SIZE, std::size_t DIM>
class multiarray  : public multiarray_mutable<multiarray<T,SIZE,DIM>> {
	std::vector<T> data; //We use a vector (even though it is on dynamic memory) because of move semantics. It will be faster to return.	

public:
	static constexpr std::size_t size = SIZE;
	static constexpr std::size_t dimensions = DIM;
	using value_type = T;
	using index_type = std::array<std::size_t,dimensions>;
private:
	static constexpr std::size_t power(std::size_t dim) { return (dim<=0)?1:SIZE*power(dim-1); }
	static constexpr std::size_t index_of(const index_type& indices) {
		std::size_t r = 0;
		for (std::size_t i = 0; i<DIM;++i) r+=indices[i]*power(i);
		return r;
	}
public:
	multiarray() : data(power(DIM))              {}
	multiarray(const T& t) : data(power(DIM),t)  {}
	multiarray(const multiarray& that) = default;
	multiarray(multiarray&& that) noexcept = default;
	//In C++ we trust for the automatic copy and move constructors, and automatic copy and move assignments. We might need to define them if we add new ones
	
	T& operator[](const index_type& indices) { return data[index_of(indices)]; }
	const T& operator[](const index_type& indices) const { return data[index_of(indices)]; }
	
	template<typename M>
	multiarray& operator=(const multiarray_const<M>& that) {
		static_assert(std::is_convertible<value_type,typename M::value_type>::value,"Should have the same value type");
		static_assert(M::dimensions == dimensions,"Should have the same dimensions");
		static_assert(M::size == size,"Should have the same dimensions");

		std::array<std::size_t,dimensions> resolution; resolution.fill(SIZE);
		for (auto p : multidimensional_range(resolution)) 
			(*this)[p] = static_cast<const M&>(that)[p];
		return (*this);
	}
	template<typename M>
	multiarray(const multiarray_const<M>& that) {
		static_assert(std::is_convertible<value_type,typename M::value_type>::value,"Should have the same value type");
		static_assert(M::dimensions == dimensions,"Should have the same dimensions");
		static_assert(M::size == size,"Should have the same dimensions");
		std::array<std::size_t,dimensions> resolution; resolution.fill(SIZE);
		for (auto p : multidimensional_range(resolution)) 
			(*this)[p] = static_cast<const M&>(that)[p];
	}

	multiarray& operator=(multiarray&& that) noexcept = default;
	

	T& at(const std::array<std::size_t,DIM>& indices) {
		if (!range_check(indices)) throw std::out_of_range("Index out of range in multidimensional array");
		return data.at(index_of(indices));
	}
};

template<typename M>
multiarray<typename M::value_type, M::size, M::dimensions> clone(const multiarray_const<M>& m) {
    multiarray<typename M::value_type, M::size, M::dimensions> sol;
    sol=m;
    return sol;
}
