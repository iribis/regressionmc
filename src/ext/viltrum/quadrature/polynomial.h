#pragma once

#include "../multiarray/multiarray.h"
#include "../multiarray/array.h"
#include "range.h"
#include "../quadrature/multidimensional-range.h"

template<std::size_t N>
struct horner {
	template<typename Float, typename T, std::size_t DIM>
	static auto evaluate(const Float& t, const std::array<T,DIM>& c) {
		if constexpr (N==DIM-1) return c[N];
		else return c[N] + t*horner<N+1>::evaluate(t,c);
	}
	
	template<typename Float, typename T, std::size_t DIM>
	static auto evaluate_integral(const Float& t, const std::array<T,DIM>& c) {
		if constexpr (N==DIM-1) return c[N]/(N+1);
		else if constexpr (N==0) return t*(c[0] + t*horner<1>::evaluate_integral(t,c));
		else return c[N]/(N+1) + t*horner<N+1>::evaluate_integral(t,c);
	}
};


template<typename T, typename Float, std::size_t SIZE, std::size_t DIM>
struct Polynomial {
    multiarray<T,SIZE,DIM> coefficients_;
	Range<Float,DIM> range_;
	
	//IT WOULD BE FANTASTIC IF ALL THESE OPERATIONS COULD BE DONE WITHOUT
	// operator+= and operator *=
	template<typename MA>
	T eval(const MA& ma, const std::array<Float,DIM>& x) const {
		constexpr std::size_t N = (MA::dimensions-1);
		auto f = [&] (const auto& c) -> T {
				return horner<0>::evaluate(x[N],c);
			};
        if constexpr (MA::dimensions > 1) return eval(ma.fold(f,N),x);
		else return ma.fold(f,N).value(); 
	}
	
	template<typename MA, std::size_t DIMSUB>
	T eval_integral(const MA& ma, const std::array<Float,DIMSUB>& a, 
								  const std::array<Float,DIMSUB>& b) const {
		constexpr std::size_t N = (MA::dimensions-1);
		if constexpr (MA::dimensions > DIMSUB) {
			auto f = [&] (const auto& c) -> T {
				return horner<0>::evaluate_integral(Float(1),c);
			};
			return eval_integral(ma.fold(f,N),a,b);
		}
		else {		
			auto f = [&] (const auto& c) -> T {
					return horner<0>::evaluate_integral(b[N],c)-
						horner<0>::evaluate_integral(a[N],c);
			};

			if constexpr (MA::dimensions > 1) return eval_integral(ma.fold(f,N),a,b);
			else return ma.fold(f,N).value();
		}
	}
	
	template<std::size_t DIMSUB, typename MA>
	auto precalculated_integral_dimension(const MA& ma) const {
		std::cerr<<MA::dimensions<<" vs "<<DIMSUB<<std::endl;
		if constexpr (MA::dimensions > DIMSUB) {
			auto f = [&] (const auto& c) -> T {
				return horner<0>::evaluate_integral(Float(1),c);
			};
			return this->template precalculated_integral_dimension<DIMSUB>(ma.fold(f,MA::dimensions-1));
		} else {
			return Polynomial<T,Float,SIZE,DIMSUB>(ma,
				Range<Float,DIMSUB>(resize<DIMSUB>(range().min()),resize<DIMSUB>(range().max())));
		}
	}
public:
	const Range<Float,DIM>& range() const { return range_; }
	const multiarray<T,SIZE,DIM>& coefficients() const { return coefficients_; }
    const T& coefficients(const std::array<Float,DIM>& i) const { return coefficients()[i]; }

    Polynomial(multiarray<T,SIZE,DIM>&& coefficients, Range<Float,DIM>&& range) :
		coefficients_(std::forward<multiarray<T,SIZE,DIM>>(coefficients)),
		range_(std::forward<Range<Float,DIM>>(range)) {}
    Polynomial(multiarray<T,SIZE,DIM>&& coefficients, const Range<Float,DIM>& range) :
		coefficients_(std::forward<multiarray<T,SIZE,DIM>>(coefficients)),
		range_(range) {}
	
    T operator()(const std::array<Float,DIM>& x) const {
		return eval(coefficients(), range().pos_in_range(x));
    }
	
	template<std::size_t DIMSUB>
	T integral(const std::array<Float,DIMSUB>& a, const std::array<Float,DIMSUB>& b) const {
		return range().volume()*eval_integral(coefficients(), range().pos_in_range(a), range().pos_in_range(b));
    }
	
	template<std::size_t DIMSUB>
	T integral(const Range<Float,DIMSUB>& r) const {
		return integral(r.min(),r.max());
    }
	
	
	T integral() const {
		//return range().volume()*eval_integral(coefficients(), std::array{Float(0)}, std::array{Float(1)});
		return range().volume()*eval_integral(coefficients(), std::array<Float, 1>{Float(0)}, std::array<Float,1>{Float(1)});
    }
	
	template<std::size_t DIMSUB>
	Polynomial<T,Float,SIZE,DIMSUB> precalculated_integral() const {
		return this->template precalculated_integral_dimension<DIMSUB>(
			this->coefficients());
	}
};
