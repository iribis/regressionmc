#pragma once

#include "slice.h"
#include "array.h"




namespace detail {
template<typename F, typename MA>
typename std::enable_if<!std::is_convertible<F,typename MA::value_type>::value,void>::type fill(const F& f, MA& ma) {
	for (std::size_t i = 0; i<MA::size; ++i)
		if constexpr (MA::dimensions == 1) 
			ma[{i}] = f(std::array<float,1>{{float(i)/float(MA::size-1)}});
		else {
			auto s = ma.slice(i);
			detail::fill([f,i] (const std::array<float,MA::dimensions-1>& p) {
				return f( (float(i)/float(MA::size-1)) | p); },s);
		}
}
template<typename V, typename MA>
typename std::enable_if<std::is_convertible<V,typename MA::value_type>::value,void>::type fill(const V& v, MA& ma) {
	for (std::size_t i = 0; i<MA::size; ++i)
		if constexpr (MA::dimensions == 1) 
			ma[{i}] = v;
		else {
			auto s = ma.slice(i);
			detail::fill(v,s);
		}
}

/*
template<typename F, typename MA, typename MAS>
void fill_blank(const F& f, MA& ma, const MAS& mas, std::size_t dim, std::size_t part, std::size_t out_of) {
	static_assert((MA::dimensions == MAS::dimensions) && (MA::size == MAS::size),"Incompatible multiarrays");
	std::size_t start = (MA::size-1)*(part-1);
	if constexpr (MA::dimensions == 1) {
		for (std::size_t i = 0; i<MA::size; ++i) {
			if (dim == 0) {
				if (((start+i)%out_of)==0) ma[{i}]=mas[{(start+i)/out_of}];
				else ma[{i}]=f(std::array<double,1>{{double(start+i)/double((MA::size-1)*out_of)}});
			} else ma[{i}] = mas[{i}];				
		}
	} else {
		multiarray_view<std::decay_t<MA>> vm(ma);
		for (std::size_t i = 0; i<MA::size; ++i) {
			vm.set_index_at_0(i);
			detail::fill([f,i] (const std::array<double,MA::dimensions-1>& p) {
				std::array<double,MA::dimensions> pinside; pinside[0]=double(i)/double(MA::size-1);
				std::copy(p.begin(),p.end(),pinside.begin()+1);
				return f(pinside); }, vm);
		}		
	}		
}
*/

/*
template<typename F, std::size_t SIZE, unsigned int DIM>
auto sample(const F& f) -> multiarray<decltype(f(std::declval<std::array<double,DIM>>())),SIZE,DIM> {
	multiarray<decltype(f(std::declval<std::array<double,DIM>>())),SIZE,DIM> ma;
	fill(f,ma);
	return ma;
}
*/

}
