#pragma once

#include "multiarray-constref.h"


//F is a function that gets an array of size SIZE and returns a single element.
//This class is not meant to be used directly.
template<typename F, typename Base, std::size_t DIM = Base::dimensions - 1> //We need a particular case when DIM becomes 0
class multiarray_folded : public multiarray_const<multiarray_folded<F,Base,DIM>> {
	F f;  
	Base base; 
	std::size_t dim_to_fold;
public:
	static constexpr std::size_t size = Base::size;
	static constexpr std::size_t dimensions = DIM;
	using value_type = decltype(std::declval<F>()(std::declval<std::array<typename Base::value_type,Base::size>>()));
	using index_type = std::array<std::size_t,dimensions>;


	multiarray_folded(F&& f, Base&& base, std::size_t dim_to_fold = 0) : f(std::forward<F>(f)), base(std::forward<Base>(base)),dim_to_fold(dim_to_fold) { }
	multiarray_folded(const F& f, Base&& base, std::size_t dim_to_fold = 0) : f(f), base(std::forward<Base>(base)),dim_to_fold(dim_to_fold) { }
	
	//In C++ we trust for the automatic copy and move constructors, and automatic copy and move assignments.
	value_type operator[](const index_type& indices) const { 
		std::array<std::size_t,DIM+1> base_indices;
		unsigned int i,ibase;
		for (i=0,ibase=0;i<DIM;++i, ++ibase) {
			if (i==dim_to_fold) ++ibase;
			base_indices[ibase]=indices[i];
		}
		
		std::array<typename Base::value_type,size> values;
		for (std::size_t e=0;e<size;++e) { base_indices[dim_to_fold] = e; values[e]=base[base_indices]; }
		return f(values);
	}
};


//F is a function that gets an array of size SIZE and returns a single element.
//This class is not meant to be used directly.
template<typename F, typename Base> //We need a particular case when DIM becomes 0
class multiarray_folded<F,Base,0> {
	F f;  
	Base base; 
	std::size_t dim_to_fold;
public:
	static constexpr std::size_t size = Base::size;
	static constexpr int dimensions   = 0;
	using value_type = decltype(std::declval<F>()(std::declval<std::array<typename Base::value_type,Base::size>>()));


	multiarray_folded(F&& f, Base&& base, std::size_t dim_to_fold = 0) : f(std::forward<F>(f)), base(std::forward<Base>(base)),dim_to_fold(dim_to_fold) { }
	multiarray_folded(const F& f, Base&& base, std::size_t dim_to_fold = 0) : f(f), base(std::forward<Base>(base)),dim_to_fold(dim_to_fold) { }
	
	value_type value() const {
		std::array<std::size_t,1> base_indices;
		std::array<typename Base::value_type,size> values;
		for (std::size_t e=0;e<size;++e) { base_indices[0] = e; values[e]=base[base_indices]; }
		return f(values);		
	}
	
	operator value_type() const {
		return value();
	}
	
	template<typename F2>
	auto fold(F2&& f, unsigned int d = 0) const {
		return value();
	}
	
	template<typename F2>
	auto fold_all(F2&& f) const {
		return value();
	}
};

namespace detail {
template<typename Base, typename F>
auto fold(Base&& base, F&& f, std::size_t  d) {
	return multiarray_folded<std::decay_t<F>,std::decay_t<Base>>(std::forward<F>(f),std::forward<Base>(base),d);
}

template<typename Base, typename F>
auto fold(Base& base, F&& f, std::size_t  d) {
	return multiarray_folded<std::decay_t<F>,multiarray_constref<std::decay_t<Base>>>(std::forward<F>(f),multiarray_constref<std::decay_t<Base>>(base),d);
}

template<typename Base, typename F>
auto fold_all(Base&& base, F&& f);

template<typename Base, typename F, std::size_t  DIM = std::decay_t<Base>::dimensions>
struct fold_all_helper {
	static auto apply(Base&& base, F&& f) {
		return detail::fold_all(detail::fold(std::forward<Base>(base),f),std::forward<F>(f));
	}
};

template<typename Base, typename F>
struct fold_all_helper<Base,F,0> {
	static auto apply(Base&& base, F&& f) {
		return base.value(); 
	}
};

template<typename Base, typename F>
auto fold_all(Base&& base, F&& f) {
	return detail::fold_all_helper<Base,F>::apply(std::forward<Base>(base),std::forward<F>(f));
}
}

