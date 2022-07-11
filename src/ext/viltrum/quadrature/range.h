#pragma once

#include <array>

template<typename T, std::size_t DIM>
class Range : public std::array<std::array<T,DIM>,2> {
	T _volume;
public:
	Range(const std::array<T,DIM>& a, const std::array<T,DIM>& b) :
		std::array<std::array<T,DIM>,2>{a,b}  {
		_volume = T(1);
		for (std::size_t i = 0; i<DIM; ++i) _volume*=(b[i]-a[i]);
	}
	const std::array<T,DIM>& min() const { return (*this)[0]; }
	T min(std::size_t i) const { return min()[i]; }
	const std::array<T,DIM>& max() const { return (*this)[1]; }
	T max(std::size_t i) const { return max()[i]; }
	T volume() const { return _volume; }

    bool is_inside(const std::array<T,DIM>& x) const {
        bool is = true;
        for (std::size_t i = 0; (i<DIM) && is; ++i)
           is = ( (x[i]>=min(i)) && (x[i]<=max(i)) );
        return is;
    }
	
	template<std::size_t DIMSUB>
	std::array<T,DIMSUB> pos_in_range(const std::array<T,DIMSUB>& pos) const {
		static_assert(DIMSUB<=DIM,"pos_in_range with too big dimensionsal index");
		std::array<T,DIMSUB> prange;
		for (std::size_t i = 0; i<DIMSUB; ++i) 
			prange[i] = (pos[i]-min(i))/(max(i) - min(i));
		return prange;
	}
	
	Range<T,DIM> subrange_dimension(std::size_t dim, T a, T b) const {
		std::array<T,DIM> new_a = min(); new_a[dim]=a;
		std::array<T,DIM> new_b = max(); new_b[dim]=b;
		return Range<T,DIM>(new_a, new_b);
	}

    template<std::size_t DIMSUB>
    Range<T,DIMSUB> intersection(const Range<T,DIMSUB>& that,
            std::enable_if_t<DIM >= DIMSUB>* p = nullptr) const {
		std::array<T,DIMSUB> a, b; 
        for (std::size_t d = 0; d<DIMSUB; ++d) {
            a[d] = std::max(this->min(d),that.min(d));
            b[d] = std::max(a[d],std::min(this->max(d),that.max(d))); //std::max watches out for empty ranges
        }
        return Range<T,DIMSUB>(a,b);
    }

    template<std::size_t DIMSUB>
    Range<T,DIM> intersection(const Range<T,DIMSUB>& that,
            std::enable_if_t<DIM < DIMSUB>* p = nullptr) const {
		std::array<T,DIM> a, b; 
        for (std::size_t d = 0; d<DIM; ++d) {
            a[d] = std::max(this->min(d),that.min(d));
            b[d] = std::max(a[d],std::min(this->max(d),that.max(d))); //std::max watches out for empty ranges
        }
        return Range<T,DIM>(a,b);
    }
	
    template<std::size_t DIMSUB>
    Range<T,DIM> intersection_large(const Range<T,DIMSUB>& that,
            std::enable_if_t<DIM >= DIMSUB>* p = nullptr) const {
		std::array<T,DIM> a, b; 
        for (std::size_t d = 0; d<DIMSUB; ++d) {
            a[d] = std::max(this->min(d),that.min(d));
            b[d] = std::max(a[d],std::min(this->max(d),that.max(d))); //std::max watches out for empty ranges
        }
		for (std::size_t d = DIMSUB; d<DIM; ++d) {
			a[d] = this->min(d);
			b[d] = this->max(d);
		}
        return Range<T,DIM>(a,b);
    }

    template<std::size_t DIMSUB>
    Range<T,DIMSUB> intersection_large(const Range<T,DIMSUB>& that,
            std::enable_if_t<DIM < DIMSUB>* p = nullptr) const {
		std::array<T,DIMSUB> a, b; 
        for (std::size_t d = 0; d<DIM; ++d) {
            a[d] = std::max(this->min(d),that.min(d));
            b[d] = std::max(a[d],std::min(this->max(d),that.max(d))); //std::max watches out for empty ranges
        }
		for (std::size_t d = DIM; d<DIMSUB; ++d) {
			a[d] = that.min(d);
			b[d] = that.max(d);
		}
        return Range<T,DIMSUB>(a,b);
    }
};

template<typename T, std::size_t DIM>
Range<T,DIM> range(const std::array<T,DIM>& a, const std::array<T,DIM>& b) {
    return Range<T,DIM>(a,b);
}

template<typename T>
Range<T,1> range(const T& a, const T& b) {
    return Range<T,1>(std::array<T,1>{a},std::array<T,1>{b});
}

template<typename T>
Range<T,2> range(const T& a0, const T& a1, const T& b0, const T& b1) {
    return Range<T,2>(std::array<T,2>{a0,a1},std::array<T,2>{b0,b1});
}

template<typename T>
Range<T,3> range(const T& a0, const T& a1, const T& a2, const T& b0, const T& b1, const T& b2) {
    return Range<T,3>(std::array<T,3>{a0,a1,a2},std::array<T,3>{b0,b1,b2});
}


template<typename T>
Range<T,4> range(const T& a0, const T& a1, const T& a2, const T& a3, const T& b0, const T& b1, const T& b2, const T& b3) {
    return Range<T,4>(std::array<T,4>{a0,a1,a2,a3},std::array<T,4>{b0,b1,b2,b3});
}
