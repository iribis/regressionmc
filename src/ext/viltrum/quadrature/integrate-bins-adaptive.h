#pragma once

#include <vector>
#include "integrate-bins-stepper.h"

template<typename Nested, typename Error>
class StepperBinsAdaptive {
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
            else for (auto pos : multidimensional_range(start_bin, end_bin)) {
                std::array<Float, DIMBINS> submin, submax;
//                Range<Float,DIM> subrange = range;
                for (std::size_t i=0;i<DIMBINS;++i) {
//                    subrange = subrange.subrange_dimension(i,range.min(i)+pos[i]*drange[i],range.min(i)+(pos[i]+1)*drange[i]);
                    submin[i] = range.min(i)+pos[i]*drange[i];
                    submax[i] = range.min(i)+(pos[i]+1)*drange[i];
                }
                //The commented stuff should work and does not, we will have to solve it
//                bins(pos) += factor*r.integral_subrange(subrange.intersection(r.range()));
                bins(pos) += factor*r.integral_subrange(Range<Float,DIMBINS>(submin,submax).intersection(r.range()));
            }
        }
    }

    StepperBinsAdaptive(Nested&& nested, Error&& error) : adaptive(std::forward<Nested>(nested), std::forward<Error>(error)) { }
};

template<typename Nested, typename Error>
auto stepper_bins_adaptive(Nested&& nested, Error&& error) {
    return StepperBinsAdaptive<std::decay_t<Nested>,std::decay_t<Error>>(std::forward<Nested>(nested), std::forward<Error>(error));
}

template<typename N>
auto stepper_bins_adaptive(N&& nested) {
    return stepper_bins_adaptive(std::forward<N>(nested), error_single_dimension_standard());
}





