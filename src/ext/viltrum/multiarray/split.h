#pragma once

#include "slice.h"
#include "fill.h"
#include "multiarray.h"
#include <list>

namespace detail {

template<typename F, typename MA>
std::vector<multiarray<typename MA::value_type,MA::size,MA::dimensions>>
split(const F& f, const MA& ma, std::size_t dim, std::size_t parts) {
	assert(dim < MA::dimensions);
	std::vector<multiarray<typename MA::value_type,MA::size,MA::dimensions>> s(parts);
	std::size_t full_size = parts*(MA::size-1) + 1;
	for (std::size_t i = 0; i<full_size; ++i) {
		std::size_t part = i/(MA::size-1);
		std::size_t position = i - part*(MA::size-1);
		float v = float(i)/float(full_size - 1);
		if (part>=parts) //Particular case for the last one
		{ --part; position+=(MA::size-1); }
		if ( (i%parts) == 0)  {//We copy the values
			if constexpr (MA::dimensions > 1) 
				s[part].slice(position,dim) = ma.slice(i/parts,dim);
			else
				s[part][{position}] = ma[{i/parts}];
		} else {
			if constexpr (MA::dimensions > 1)
				s[part].slice(position,dim).fill(
					[f,dim,v] 
						(const std::array<float,MA::dimensions-1>& p) {
							return f(insert(p,v,dim));
					});
			else
				s[part][{position}] = f(std::array<float,1>{v});
		}
		//We copy at the merging points (limits between two splitted multiarrays)
		if ((position == 0) && (part > 0)) {
			if constexpr (MA::dimensions > 1)
				s[part-1].slice(MA::size-1,dim) = s[part].slice(0,dim);
			else
				s[part-1][{MA::size-1}]=s[part][{0}];	
		}
	}
	return s;
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
