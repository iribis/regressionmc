#pragma once

#include "multiarray-crtp.h"

//Class to avoid copies in other multiarray classes.
template<typename Base>
class multiarray_constref : public multiarray_const<multiarray_constref<Base>> {
	const Base& base;
public:
	static constexpr std::size_t size = Base::size;
	static constexpr int dimensions = Base::dimensions;
	using value_type = typename Base::value_type;	
	using index_type = typename Base::index_type;

	//In C++ we trust for the automatic copy and move constructors, and automatic copy and move assignments.
	multiarray_constref(const Base& base):base(base) {}
	value_type operator[](const index_type& indices) const { return base[indices]; }
};