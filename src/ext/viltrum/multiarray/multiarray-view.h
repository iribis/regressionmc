#pragma once
#include "multiarray-crtp.h"


//Class that fixes dim 0.
template<typename Base>
class multiarray_view : multiarray_mutable<multiarray_view<Base>> {
	Base& base;
	std::size_t index_at_0;
public:
	void set_index_at_0(std::size_t i) { index_at_0=i; }
	static constexpr std::size_t size = Base::size;
	static constexpr std::size_t dimensions = Base::dimensions-1;
	using value_type = typename Base::value_type;
	using index_type = std::array<std::size_t,dimensions>;
	//In C++ we trust for the automatic copy and move constructors, and automatic copy and move assignments.
	multiarray_view(Base& base,std::size_t index_at_0 = 0):base(base),index_at_0(index_at_0) {}
	value_type& operator[](const index_type& indices) { 
		std::array<std::size_t,Base::dimensions> view_indices; 
		view_indices[0]=index_at_0;
		std::copy(indices.begin(),indices.end(),view_indices.begin()+1);
		return base[view_indices]; 
	}
	const value_type& operator[](const index_type& indices) const { 
		std::array<std::size_t,Base::dimensions> view_indices; 
		view_indices[0]=index_at_0;
		std::copy(indices.begin(),indices.end(),view_indices.begin()+1);
		return base[view_indices]; 
	}
};
