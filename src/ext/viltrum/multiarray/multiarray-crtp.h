#pragma once

#include <vector>
#include <array>
#include <cassert>
#include <stdexcept>

template<typename T, std::size_t SIZE, std::size_t DIM>
class multiarray;
namespace detail {
template<typename Base, typename F>
auto fold(Base&& base, F&& f, std::size_t d=0);

template<typename Base, typename F>
auto fold(Base& base, F&& f, std::size_t d=0);

template<typename Base, typename F>
auto fold_all(Base&& base, F&& f);

template<typename Base, typename F>
auto transform(Base&& base, F&& f, std::size_t  d);

template<typename Base, typename F>
auto transform(Base& base, F&& f, std::size_t  d);

template<typename Base, typename F>
auto transform_all(Base&& base, const F& f);

/*
template<typename Base, typename F>
auto transform_all(Base& base, F&& f);
*/
template<typename F, typename MA>
typename std::enable_if<!std::is_convertible<F,typename MA::value_type>::value,void>::type fill(const F& f, MA& ma);

template<typename V, typename MA>
typename std::enable_if<std::is_convertible<V,typename MA::value_type>::value,void>::type fill(const V& v, MA& ma);

template<typename MA>
auto slice(const MA& ma, std::size_t dimension, std::size_t index) noexcept;

template<typename MA>
auto slice(MA& ma, std::size_t dimension, std::size_t index) noexcept;

template<typename F, typename MA>
std::vector<multiarray<typename MA::value_type,MA::size,MA::dimensions>>
split(const F& f, const MA& ma, std::size_t dim, std::size_t parts);
}


namespace {
template<typename MA>
bool constexpr range_check(const MA* ma, const typename MA::index_type& indices) {
	bool r = true;
	for (auto i : indices) r = (r && (i<MA::size));
	return r;
}
}


template<typename MA>
class multiarray_const {
public:
	template<typename I>
	const auto& at(const I& indices) const {
		if (!range_check(static_cast<const MA*>(this),indices)) throw std::out_of_range("Index out of range in multidimensional array");
		return static_cast<const MA*>(this)->operator[](indices);
	}
	template<typename F>
	auto fold(F&& f, unsigned int d = 0) const {
		return detail::fold(*static_cast<const MA*>(this),f,d);
	}
	template<typename F>
	auto fold_all(F&& f) const {
		return detail::fold_all(*static_cast<const MA*>(this),f);
	}

	template<typename F>
	auto transform(F&& f, unsigned int d = 0) const {
		return detail::transform(*static_cast<const MA*>(this),f,d);
	}
	template<typename F>
	auto transform_all(const F& f) const {
		return detail::transform_all(*static_cast<const MA*>(this),f);
	}
	auto slice(std::size_t index, int dimension = 0) const noexcept {
		return detail::slice(static_cast<const MA&>(*this),dimension,index);
	}
	template<typename F>
	auto split(const F& f, int dim, std::size_t parts) const {
		return detail::split(f,static_cast<const MA&>(*this),dim,parts);
	}

//	template<typename MA2>
//	auto operator==(const multiarray_const<MA2>& ma2) const ->
//		std::enable_if_t<
};

template<typename MA>
class multiarray_mutable : public multiarray_const<MA> {
public:
	template<typename I>
	auto& at(const I& indices) {
		if (!range_check(this,indices)) throw std::out_of_range("Index out of range in multidimensional array");
		return static_cast<MA*>(this)->operator[](indices);
	}
	//We need to redefine this because otherwise it gets hidden
	template<typename I>
	const auto& at(const I& indices) const {
		if (!range_check(this,indices)) throw std::out_of_range("Index out of range in multidimensional array");
		return static_cast<const MA*>(this)->operator[](indices);
	}
	template<typename F, typename Float = double>
	void fill(const F& f) {
		detail::fill(f,*static_cast<MA*>(this));
	}
	//We need to redefine this because otherwise it gets hidden
	auto slice(std::size_t index, int dimension = 0) const noexcept {
		return detail::slice(static_cast<const MA&>(*this),dimension,index);
	}
	auto slice(std::size_t index, int dimension = 0) noexcept {
		return detail::slice(static_cast<MA&>(*this),dimension,index);
	}
};

#define ASSIGNMENT(MThis) \
template<typename MThat> \
MThis& operator=(const multiarray_const<MThat>& that) { \
	static_assert(dimensions == MThat::dimensions, "Assignment of multiarrays with different number of dimensions"); \
	static_assert(size == MThat::size, "Assignment of multiarrays with different sizes"); \
	if constexpr(dimensions == 1) { \
		for (std::size_t i = 0; i < size; ++i) \
			(*this)[{i}] = static_cast<const MThat&>(that)[{i}]; \
	} else { \
		for (std::size_t i = 0; i < size; ++i) \
			this->slice(i) = that.slice(i); \
	} \
	return (*this); \
}  \
MThis& operator=(const MThis& that) { \
	if constexpr(dimensions == 1) { \
		for (std::size_t i = 0; i < size; ++i) \
			(*this)[{i}] = that[{i}]; \
	} else { \
		for (std::size_t i = 0; i < size; ++i) \
			this->slice(i) = that.slice(i); \
	} \
	return (*this); \
}  
