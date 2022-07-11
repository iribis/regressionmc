#pragma once
#include <array>
#include <algorithm>

template<typename T, std::size_t N>
auto operator|(const std::array<T,N>& a, const T& t) noexcept -> std::array<T,N+1> {
	std::array<T,N+1> s;
	std::copy(a.begin(),a.end(),s.begin());
	s[N]=t;
	return s;
}

template<typename T, std::size_t N>
auto operator|(const T& t, const std::array<T,N>& a) noexcept -> std::array<T,N+1> {
	std::array<T,N+1> s;
	std::copy(a.begin(),a.end(),s.begin()+1);
	s[0]=t;
	return s;
}

template<typename T, std::size_t N1, std::size_t N2>
auto operator|(const std::array<T,N1>& a1,const std::array<T,N2>& a2) noexcept -> std::array<T,N1+N2> {
	std::array<T,N1+N2> s;
	std::copy(a1.begin(),a1.end(),s.begin());
	std::copy(a2.begin(),a2.end(),s.begin()+N1);
	return s;
}

template<typename T, std::size_t N>
auto insert(const std::array<T,N>& a, const T& t, std::size_t at) noexcept -> std::array<T,N+1> {
	std::array<T,N+1> s;
	std::copy(a.begin(),a.begin()+at,s.begin());
	s[at]=t;
	std::copy(a.begin()+at,a.end(),s.begin()+at+1);	
	return s;
}

template<std::size_t NNEW, std::size_t N>
std::array<float,NNEW> resize(const std::array<float,N>& a) {
    std::array<float,NNEW> s;
    for (std::size_t i = 0; i<NNEW; ++i) s[i] = a[std::min(i,N-1)];
    return s;
} 