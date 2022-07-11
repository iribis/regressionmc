#pragma once

#include <array>
#include "../multiarray/multiarray.h"
#include "../multiarray/fill.h"
#include "../multiarray/fold.h"
#include "../multiarray/transform.h"
#include "../multiarray/io.h"
#include "../multiarray/split.h"
#include "polynomial.h"
#include "range.h"
#include "rules.h"
#include "nested.h"
#include <cmath>


template<typename Float, typename Q, std::size_t DIM, typename VT>
class Region {
	Q quadrature; // Can be neasted quadrature
	Range<Float, DIM> _range; // Certainly the range represented by this region
	
public:
	using value_type = VT;
	static constexpr std::size_t dimensions = DIM;
	const Range<Float, DIM>& range() const {
		return _range;
	}
	
private:
	multiarray<value_type,Q::samples,DIM> data; // This store the data
	
	// Certainly an helper to rescale the call to be in the good range
	// for f
	template<typename F>
	constexpr auto f_in_range(const F& f) const { 
		return([&] (const std::array<Float,DIM>& p) {
			std::array<Float,DIM> prange;
			for (std::size_t i = 0; i<DIM; ++i) 
				prange[i] = p[i]*(range().max(i) - range().min(i)) + range().min(i);
			return f(prange);
		}); 
	}
	
	constexpr Float volume_from(std::size_t start) const {
		Float v(1); 
		for (std::size_t i = start; i<DIM; ++i) v*=std::abs(range().max(i) - range().min(i));
		return v;	
	}
public:
	Region(const Q& q, 
			const std::array<Float, DIM>& range_min, 
			const std::array<Float, DIM>& range_max,
			multiarray<value_type,Q::samples,DIM>&& d) : 
				quadrature(q), _range(range_min, range_max), 
				data(std::forward<multiarray<value_type,Q::samples,DIM>>(d)) { }

	template<typename F>
	Region(const F& f, const Q& q, 
			const std::array<Float, DIM>& range_min, 
			const std::array<Float, DIM>& range_max) : 
				quadrature(q), _range(range_min, range_max) {
		data.fill(this->f_in_range(f));
	}

    const Q& quadrature_rule() const { return quadrature; }
	
	value_type integral() const { return range().volume()*data.fold_all(quadrature); }
	
private:
	//Allegedly starts at 0
	template<typename MA, std::size_t DIMSUB>
	value_type app_at(const MA& ma, const std::array<Float,DIMSUB>& pos, 
			std::enable_if_t<MA::dimensions <= (DIM-DIMSUB+1)>* p = nullptr) const {
		// Does it is the last dimension? if so, continue on fold_all until nothing remains (integrate)
		return ma.fold([&] (const auto& v) 
				{ return quadrature.at(pos[DIMSUB - 1],v);}).fold_all(quadrature)*volume_from(DIMSUB);
	}
	
	// This reduce the dimension of the array at each call
	// DIM: Size of the domain, DIMSUB -> query size
	template<typename MA, std::size_t DIMSUB>
	auto app_at(const MA& ma, const std::array<Float,DIMSUB>& pos, 
			std::enable_if_t<MA::dimensions >= ((DIM-DIMSUB)+2)>* p = nullptr) const {
		return app_at(ma.fold([&] (const auto& v) 
				{ return quadrature.at(pos[DIMSUB - MA::dimensions],v); }),pos);
	}

public:
	//If you do not reach the end it integrates in the other dimensions
	template<std::size_t DIMSUB>
	value_type approximation_at(const std::array<Float,DIMSUB>& pos) const {
		static_assert(DIM>=DIMSUB,"Cannot approximate with bigger number of dimensions than the region");
		return app_at(data,range().pos_in_range(pos));
	}
	
	//If you do not reach the end it integrates in the other dimensions
	value_type approximation_at(Float pos) const {
		return approximation_at(std::array<Float,1>{pos});
	}
	
	Polynomial<value_type,Float,Q::samples,DIM> polynomial() const {
		return Polynomial<value_type,Float,Q::samples,DIM>(
			clone(data.transform_all([&] (const std::array<value_type,Q::samples>& c, std::size_t i) -> value_type { return quadrature.coefficient(c,i); })),
			range());
	}
	
private:
	//Starts from 0
	template<typename MA, std::size_t DIMSUB>
	value_type sub_first(const MA& ma,  
			const std::array<Float,DIMSUB>& a, const std::array<Float,DIMSUB>& b, 
			std::enable_if_t<MA::dimensions <= (DIM-DIMSUB+1)>* p = nullptr) const {
		return ma.fold([&] (const auto& v) 
			{ return quadrature.subrange(a[DIMSUB - 1], b[DIMSUB - 1],v); }).fold_all(quadrature);
	}
	
	//Allegedly starts at 0
	template<typename MA, std::size_t DIMSUB>
	auto sub_first(const MA& ma, 
			const std::array<Float,DIMSUB>& a, const std::array<Float,DIMSUB>& b, 
			std::enable_if_t<MA::dimensions >= ((DIM-DIMSUB)+2)>* p = nullptr) const {
		return sub_first(ma.fold([&] (const auto& v) 
			{ return quadrature.subrange(a[DIMSUB - MA::dimensions], b[DIMSUB- MA::dimensions],v);}),a,b);
	}
public:
	template<std::size_t DIMSUB>
	value_type integral_subrange_first(const std::array<Float,DIMSUB>& a, 
						const std::array<Float,DIMSUB>& b) const {
		// Make sense the assert
		static_assert(DIM>=DIMSUB,"Cannot calculate the subrange integral for that many dimensions, as the region has less dimensions");
		return range().volume()*sub_first(data,range().pos_in_range(a),range().pos_in_range(b));
	}

private:
	template<typename MA, std::size_t DIMSUB>
	value_type sub_last(const MA& ma, 
			const std::array<Float,DIMSUB>& a, const std::array<Float,DIMSUB>& b) const {
        if constexpr (MA::dimensions > DIMSUB)
            return sub_last(ma.fold(quadrature,DIMSUB),a,b);
        else if constexpr (MA::dimensions > 1)
            return sub_last(ma.fold([&] (const auto& v) { return quadrature.subrange(a[MA::dimensions-1],b[MA::dimensions-1],v); },MA::dimensions-1),a,b);
        else
            return ma.fold([&] (const auto& v) { return quadrature.subrange(a[0],b[0],v); }).value();
	}
public:
	template<std::size_t DIMSUB>
	value_type integral_subrange_last(const std::array<Float,DIMSUB>& a, 
						const std::array<Float,DIMSUB>& b) const {
		static_assert(DIM>=DIMSUB,"Cannot calculate the subrange integral for that many dimensions, as the region has less dimensions");
		return range().volume()*sub_last(data,range().pos_in_range(a),range().pos_in_range(b));
    }

   
	template<std::size_t DIMSUB>
	value_type integral_subrange(const std::array<Float,DIMSUB>& a, 
						const std::array<Float,DIMSUB>& b) const {
        return integral_subrange_last(a,b);
	}

    template<std::size_t DIMSUB>
    value_type integral_subrange(const Range<Float,DIMSUB>& subrange) const {
        return integral_subrange(subrange.min(),subrange.max());
    }
	
	value_type integral_subrange(Float a, Float b) const {
		return integral_subrange(std::array<Float,1>{a},std::array<Float,1>{b});
	}
	
	// Return the region splitted (mid point)?
	template<typename F>
	std::vector<Region<Float,Q,DIM,value_type>> split(const F& f, std::size_t dimension = 0, std::size_t parts = 2) const {
		// Do the splitting for a given dimension 
		assert(dimension < DIM);

		// 2 part splitting
		auto subdatas = data.split(f_in_range(f),dimension,parts);
		std::vector<Region<Float,Q,DIM,value_type>> sol; sol.reserve(parts);
		std::array<Float,DIM> range_midmin = range().min();
		std::array<Float,DIM> range_midmax = range().max();
		Float d = (range().max(dimension) - range().min(dimension))/(Float(parts));
		for (std::size_t i = 0; i<parts; ++i) { // Ok Cool
			range_midmax[dimension] = (i<(parts-1))?(range().min(dimension)+d*(i+1)):range().max(dimension);
			sol.emplace_back(quadrature,range_midmin,range_midmax,std::move(subdatas[i]));
			range_midmin[dimension] = range_midmax[dimension];
		}
		return sol;
	}
	
	// Create split in all directions
	template<typename F>
	std::vector<Region<Float,Q,DIM,value_type>> split_all(const F& f, std::size_t parts = 2) const {
		std::vector<Region<Float,Q,DIM,value_type>> sol, accumulated;
		sol.reserve(std::size_t(std::pow(parts,DIM))); 
		sol.push_back(*this);
		for (std::size_t d = 0; d < DIM; ++d) {
			accumulated.clear();
			accumulated.reserve(sol.capacity());
			for (auto r : sol) 
				for (auto subregion : r.split(f,d,parts))
					accumulated.emplace_back(subregion);
			
			sol.swap(accumulated);
		}
		return sol;
	}
	

	// Certainly a float
	template<typename QDT = Q, typename = typename std::enable_if<is_nested<QDT>::value>::type >
	value_type error(std::size_t dim) const {
		assert(dim < DIM);

		// data.fold -> compute the error for the given dimension,
		// then fold all will just sum all to give the final result 
		return range().volume()*
			data.fold([&] (const auto& v) { return quadrature.error(v); },dim)
			    .fold_all(quadrature);
	}
	
	/*
	 * Norm :: value_type -> Float
	 */
	template<typename Norm, typename QDT = Q, typename = typename std::enable_if<is_nested<QDT>::value>::type >
	std::tuple<Float,std::size_t> max_error_dimension(const Norm& norm) const {
		// This is similar to "error" operator
		// but less customasible.
		Float max_err = 0; std::size_t max_dim = 0; Float err;
		for (std::size_t d = 0; d<DIM; ++d) {
			err = norm(this->error(d));
			if (err>max_err) {
				max_err = err; max_dim = d;
			}
		}
		return std::tuple<Float,std::size_t>(max_err,max_dim);
	}
	
	template<typename QDT = Q, typename = typename std::enable_if<is_nested<QDT>::value && std::is_arithmetic<value_type>::value>::type >
	std::tuple<Float,std::size_t> max_error_dimension() const {
		return max_error_dimension(
			[] (const value_type& v) { return Float(std::abs(v)); }
		);
	}

	template<typename QDT = Q, typename = typename std::enable_if<is_nested<QDT>::value>::type >
	value_type error() const {
		// Sum of error becomes the error of differences
		// fold_all without closure call the quadrature normally.
		return range().volume()*(data.fold_all(quadrature) - 
					   data.fold_all([&] (const auto& v) { return quadrature.low(v); }));
	}		
};

template<typename Float, typename F, typename Q, std::size_t DIM>
auto region(const F& f, const Q& q, 
			const std::array<Float, DIM>& range_min, 
			const std::array<Float, DIM>& range_max) {
	return Region<Float,Q,DIM,std::decay_t<decltype(f(range_min))>>(f,q,range_min, range_max);
}

// Extended regions: Can add additional information?
// Yes, it will be region + dimension with the highest error
template<typename R, typename E>
class ExtendedRegion : public R {
	E e;
public:
	ExtendedRegion(R&& r, E&& e) : R(std::forward<R>(r)), e(std::forward<E>(e)) { }
	ExtendedRegion(const R& r, E&& e) : R(r), e(std::forward<E>(e)) { }
	ExtendedRegion(R&& r, const E& e) : R(std::forward<R>(r)), e(e) { }
	ExtendedRegion(const R& r, const E& e) : R(r), e(e) { }
	
	const E& extra() const { return e; }
};
