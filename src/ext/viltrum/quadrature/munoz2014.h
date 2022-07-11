#pragma once

#include "../multiarray/array.h"
#include "integrate.h"
#include "integrate-bins-adaptive.h"
#include "monte-carlo.h"

class IntegratorBinsMunoz2014 {
    unsigned long spp; 
    unsigned long spp_pixel;
    double error_rate;
    std::size_t seed;
public:
    typedef void is_integrator_tag;

	template<typename Bins, typename F, typename Float, std::size_t DIMBINS, std::size_t DIM>
	void integrate(Bins& bins, const std::array<std::size_t,DIMBINS>& bin_resolution, const F& f, const Range<Float,DIM>& r) const {
        if constexpr(DIMBINS == 2 && DIM == 3) {
            integrator_bins_stepper(stepper_bins_per_bin(stepper_monte_carlo_uniform(seed)),spp_pixel).integrate(bins,bin_resolution,
                [&] (const std::array<Float,2>& x) {
                    return integrator_adaptive_iterations(nested(simpson,trapezoidal),error_single_dimension_size(error_rate),spp/spp_pixel).integrate(
                    [&] (const std::array<Float,1>& t) {
                        return f(x | t[0]);
                    },range(r.min(2),r.max(2)));
                }, range(r.min(0),r.min(1),r.max(0),r.max(1)));
        } 
    }

    IntegratorBinsMunoz2014(unsigned long spp, unsigned long spp_pixel, double error_rate, std::size_t seed = std::random_device()()) :
        spp(spp), spp_pixel(spp_pixel), error_rate(error_rate), seed(seed) {}
};

auto integrator_bins_munoz_2014(unsigned long spp, unsigned long spp_pixel, double error_rate, std::size_t seed = std::random_device()()) {
    return IntegratorBinsMunoz2014(spp,spp_pixel,error_rate,seed);
}


