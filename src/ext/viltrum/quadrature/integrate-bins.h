#pragma once

#include <vector>
#include "integrate.h"
#include "multidimensional-range.h"

template<typename IntegratorPerBin>
class IntegratorBinsPerBin {
    IntegratorPerBin bin_integrator;

public:
	template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
	void integrate(Bins& bins, const std::array<std::size_t,DIMBINS>& bin_resolution,
		const F& f, const Range<Float,DIM>& range) const {
        double factor(1);   
        for (std::size_t i = 0; i < DIMBINS; ++i) factor*=bin_resolution[i];
        std::array<Float,DIMBINS> drange;
        for (std::size_t i=0;i<DIMBINS;++i) drange[i] = (range.max(i) - range.min(i))/Float(bin_resolution[i]);
        for (auto pos : multidimensional_range(bin_resolution)) {
            Range<Float,DIM> subrange = range;
            for (std::size_t i=0;i<DIMBINS;++i)
                subrange = subrange.subrange_dimension(i,range.min(i)+pos[i]*drange[i],range.min(i)+(pos[i]+1)*drange[i]);
            bins(pos) = factor*bin_integrator.integrate(f,subrange);
        }
	}

//    IntegratorQuadrature(const Q& q) : quadrature(q) { }
    IntegratorBinsPerBin(IntegratorPerBin&& pi) : 
	    bin_integrator(std::forward<IntegratorPerBin>(pi)) { }

};

template<typename IntegratorPerBin>
auto integrator_bins_per_bin(IntegratorPerBin&& i) {
    return IntegratorBinsPerBin<std::decay_t<IntegratorPerBin>>(std::forward<IntegratorPerBin>(i));
}


template<typename IntegratorBins, typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
void integrate_bins(const IntegratorBins& integrator_bins, Bins& bins, const std::array<std::size_t,DIMBINS>& resolution, const F& function, const Range<Float,DIM>& range) {
    integrator_bins.integrate(bins,resolution, function, range);
}

template<typename T>
auto vector_bins(std::vector<T>& bins) {
    return [&bins] (const std::array<std::size_t,1>& i) -> T& { return bins[i[0]]; };
}
template<typename T>
std::array<std::size_t,1> vector_resolution(const std::vector<T>& bins) { return std::array<std::size_t,1>{bins.size()}; }

template<typename IntegratorBins, typename T, typename F, typename Float, std::size_t DIM>
void integrate_bins(const IntegratorBins& integrator_bins, std::vector<T>& bins, const F& function, const Range<Float,DIM>& range) {
    auto vb = vector_bins(bins);
    integrate_bins(integrator_bins, vb, vector_resolution(bins), function, range);
}



