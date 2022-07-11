#pragma once
#include "multiarray-crtp.h"
#include "fold.h"
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>

class array_to_string {
	std::string separator;
public:
	array_to_string(const std::string& separator = " "): separator(separator) {}
	
	template<typename T, std::size_t N>
	std::string operator()(const std::array<T,N>& a) const {
		std::stringstream sstr;
		std::for_each(a.begin(), a.end(), [&] (const T& t) { sstr<<t<<separator; });
		return sstr.str();
	}
};

template<typename MA>
std::ostream& operator<<(std::ostream& os, const multiarray_const<MA>& ma) {
	if constexpr (MA::dimensions == 1) 
		os<<ma.fold(array_to_string(" ")).value();
	else if constexpr (MA::dimensions == 2) 
		os<<ma.fold(array_to_string(" "),1).fold(array_to_string("\n")).value();
	else 
		os<<ma.fold(array_to_string(" "),1).fold(array_to_string("\n")).fold_all(array_to_string("-----------------------------\n"));
	return os;
}