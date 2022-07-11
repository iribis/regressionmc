#pragma once

#include "multiarray-constref.h"


//F is a function that gets an array of size SIZE and an index returns a value.
//This class is not meant to be used directly.
template<typename F, typename Base, std::size_t DIM = Base::dimensions> //We need a particular case when DIM becomes 0
class multiarray_transformed : public multiarray_const<multiarray_transformed<F,Base,DIM>> {
	F f;  
	Base base; 
	std::size_t dim_to_transform;
public:
	static constexpr std::size_t size = Base::size;
	static constexpr std::size_t dimensions = DIM;
	using value_type = std::decay_t<decltype(std::declval<F>()(std::declval<std::array<typename Base::value_type,Base::size>>(),0))>;
	using index_type = std::array<std::size_t,dimensions>;


	multiarray_transformed(F&& f, Base&& base, std::size_t dim_to_transform = 0) : f(std::forward<F>(f)), base(std::forward<Base>(base)),dim_to_transform(dim_to_transform) { }
	multiarray_transformed(const F& f, Base&& base, std::size_t dim_to_transform = 0) : f(f), base(std::forward<Base>(base)),dim_to_transform(dim_to_transform) { }
	
	//In C++ we trust the automatic copy and move constructors, and automatic copy and move assignments.
	value_type operator[](const index_type& indices) const { 
		std::array<std::size_t,DIM> base_indices = indices;
		std::array<typename Base::value_type,size> values;
		for (std::size_t e=0;e<size;++e) { base_indices[dim_to_transform] = e; values[e]=base[base_indices]; }
		return f(values,indices[dim_to_transform]);
	}
};


namespace detail {
template<typename Base, typename F>
auto transform(Base&& base, F&& f, std::size_t  d) {
	return multiarray_transformed<std::decay_t<F>,std::decay_t<Base>>(std::forward<F>(f),std::forward<Base>(base),d);
}

template<typename Base, typename F>
auto transform(Base& base, F&& f, std::size_t  d) {
	return multiarray_transformed<std::decay_t<F>,multiarray_constref<std::decay_t<Base>>>(std::forward<F>(f),multiarray_constref<std::decay_t<Base>>(base),d);
}

    

template<std::size_t D, std::size_t BASED>
struct transform_all_helper {
    template<typename Base, typename F>
	static auto apply(Base&& base, const F& f) {
        return transform_all_helper<D+1,BASED>::apply(transform(std::forward<Base>(base),f,D),f);
	}
};

template<std::size_t D>
struct transform_all_helper<D,D> {
    template<typename Base, typename F>
	static auto apply(Base&& base, const F& f) {
		return std::forward<Base>(base);
	}
};

template<typename Base, typename F>
auto transform_all(Base&& base, const F& f) {
	return detail::transform_all_helper<0,std::decay_t<Base>::dimensions>::apply(std::forward<Base>(base),f);
}

/*
template<typename Base, typename F>
auto transform_all(Base& base, F&& f) {
	return detail::transform_all_helper<0>::apply(multiarray_constref<std::decay_t<Base>>(base),std::forward<F>(f));
}
*/
}

