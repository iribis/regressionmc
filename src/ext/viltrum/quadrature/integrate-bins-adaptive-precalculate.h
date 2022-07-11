#pragma once

#include <vector>
#include "integrate-bins-stepper.h"

template<typename Nested, typename Error>
class StepperBinsAdaptivePrecalculate {
    StepperAdaptive<Nested,Error> adaptive;

public:
    template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
    auto init(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range) const {
        return adaptive.init(f,range);
    }

    template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename R>
    void step(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, std::vector<R>& heap) const {
        adaptive.step(f,range,heap);
    }

    template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename R>
    void integral(Bins& bins, const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, const std::vector<R>& regions) const {
        std::array<Float,DIMBINS> drange;
        for (std::size_t i=0;i<DIMBINS;++i) drange[i] = (range.max(i) - range.min(i))/Float(resolution[i]);
        double factor = 1;
        for (std::size_t i=0;i<DIMBINS;++i) factor*=resolution[i];

        for (const auto& r : regions) {
            std::array<std::size_t,DIMBINS> start_bin, end_bin;
            for (std::size_t i = 0; i<DIMBINS;++i) {
                start_bin[i] = std::max(std::size_t(0),std::size_t((r.range().min(i) - range.min(i))/drange[i]));
                end_bin[i]   = std::min(resolution[i],std::size_t((r.range().max(i) - range.min(i))/drange[i])+1);
            }
            if (start_bin == end_bin) bins(start_bin)+=r.integral();
            else {
				auto polynomial = r.polynomial();
				for (auto pos : multidimensional_range(start_bin, end_bin)) {
					std::array<Float, DIMBINS> submin, submax;
					for (std::size_t i=0;i<DIMBINS;++i) {
						submin[i] = range.min(i)+pos[i]*drange[i];
						submax[i] = range.min(i)+(pos[i]+1)*drange[i];
					}
					bins(pos) += factor*polynomial.integral(Range<Float,DIMBINS>(submin,submax).intersection(r.range()));
				}
            }
        }
    }

    StepperBinsAdaptivePrecalculate(Nested&& nested, Error&& error) : adaptive(std::forward<Nested>(nested), std::forward<Error>(error)) { }
};

template<typename Nested, typename Error>
auto stepper_bins_adaptive_precalculate(Nested&& nested, Error&& error) {
    return StepperBinsAdaptivePrecalculate<std::decay_t<Nested>,std::decay_t<Error>>(std::forward<Nested>(nested), std::forward<Error>(error));
}

template<typename N>
auto stepper_bins_adaptive_precalculate(N&& nested) {
    return stepper_bins_adaptive_precalculate(std::forward<N>(nested), error_single_dimension_standard());
}





