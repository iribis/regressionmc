#pragma once

#include "integrate-bins-adaptive.h"
#include "monte-carlo.h"
#include "sample-vector.h"

template<typename Nested, typename Error, typename ResidualStepper, typename VectorSampler>
class StepperBinsAdaptiveStratifiedControlVariatesPrecalculate {
	StepperBinsAdaptive<Nested,Error> cv_stepper;
	ResidualStepper residual_stepper;
	VectorSampler vector_sampler;
	unsigned long adaptive_iterations;

    template<typename R, typename ResData, typename Float, std::size_t DIM>
    struct BinData {
        ResData residual_data;
        std::vector<const R*> regions;
        std::vector<Range<Float,DIM>> regions_subrange;
        using Sampler = decltype(std::declval<VectorSampler>()(std::declval<std::vector<R*>>()));
        Sampler sampler;
        BinData() { }
    };
    
	template<typename R, typename Float, std::size_t DIM, std::size_t DIMBINS, typename ResData>
    struct Data {
		std::vector<R> regions;
        vector_dimensions<BinData<R,ResData,Float,DIM>,DIMBINS> bin_data;
        Data(std::vector<R>&& rs, const std::array<std::size_t,DIMBINS>& resolution, const Range<Float,DIM>& range) : 
			regions(std::forward<std::vector<R>>(rs)),
            bin_data(resolution) { }
    };
public:
	template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
    auto init(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range) const {
        auto regions = cv_stepper.init(resolution,f,range);
        using R = typename decltype(regions)::value_type;
        using ResData = decltype(residual_stepper.init(f,range));
        Data<R,Float,DIM,DIMBINS,ResData> data(std::move(regions),resolution,range);
        for (unsigned long i = 0; i<adaptive_iterations; ++i)
            cv_stepper.step(resolution,f,range,data.regions);

        std::array<Float,DIMBINS> drange;
        for (std::size_t i=0;i<DIMBINS;++i) drange[i] = (range.max(i) - range.min(i))/Float(resolution[i]);
        for (auto pos : multidimensional_range(resolution)) {
           Range<Float,DIM> subrange = range;
           for (std::size_t i=0;i<DIMBINS;++i)
               subrange = subrange.subrange_dimension(i,range.min(i)+pos[i]*drange[i],range.min(i)+(pos[i]+1)*drange[i]);
           for (const R& r : data.regions) {
                Range<Float,DIM> inter = subrange.intersection(r.range());
                if (inter.volume()>0) {
                    data.bin_data[pos].regions.push_back(&r);
                    data.bin_data[pos].regions_subrange.push_back(inter);
                }
           }
           data.bin_data[pos].sampler = vector_sampler(data.bin_data[pos].regions);
           data.bin_data[pos].residual_data = residual_stepper.init(f,subrange);
        }
        return data;
    }
	
	template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename R, typename ResData>
    void step(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, Data<R,Float,DIM,DIMBINS,ResData>& data) const
	{
       for (auto pos : multidimensional_range(resolution)) {
			auto [index, probability] = data.bin_data[pos].sampler.sample();
			const R* chosen_region = data.bin_data[pos].regions[index];
			chosen_region = data.bin_data[pos].regions[index];
			residual_stepper.step([&] (const std::array<Float,DIM>& x) { return (f(x) - chosen_region->approximation_at(x))/probability; },
			    data.bin_data[pos].regions_subrange[index], data.bin_data[pos].residual_data);
       }
    }
	
	template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename R,typename ResData>
    void integral(Bins& bins, const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, const Data<R,Float,DIM,DIMBINS,ResData>& data) const {
        cv_stepper.integral(bins,resolution,f,range,data.regions);
        for (auto pos : multidimensional_range(resolution))
            bins(pos) += data.bin_data.size()*residual_stepper.integral(f,range,data.bin_data[pos].residual_data);
    }
	
	StepperBinsAdaptiveStratifiedControlVariatesPrecalculate(
		Nested&& nested, Error&& error, ResidualStepper&& rs, VectorSampler&& vs, unsigned long ai) :
			cv_stepper(std::forward<Nested>(nested), std::forward<Error>(error)),
			residual_stepper(std::forward<ResidualStepper>(rs)),
			vector_sampler(std::forward<VectorSampler>(vs)),
			adaptive_iterations(ai) { }
};

template<typename Nested, typename Error, typename ResidualStepper, typename VectorSampler>
auto stepper_bins_adaptive_stratified_control_variates_precalculate(Nested&& nested, Error&& error, ResidualStepper&& residual_stepper, VectorSampler&& vector_sampler, unsigned long adaptive_iterations) {
	return StepperBinsAdaptiveStratifiedControlVariatesPrecalculate<std::decay_t<Nested>,std::decay_t<Error>,std::decay_t<ResidualStepper>,std::decay_t<VectorSampler>>(
		std::forward<Nested>(nested),std::forward<Error>(error),std::forward<ResidualStepper>(residual_stepper),std::forward<VectorSampler>(vector_sampler),adaptive_iterations);
}

template<typename Nested>
auto stepper_bins_adaptive_stratified_control_variates_precalculate(Nested&& nested, unsigned long adaptive_iterations, std::size_t seed_mc = std::random_device()(), std::size_t seed_vs = std::random_device()()) {
	return stepper_bins_adaptive_stratified_control_variates_precalculate(std::forward<Nested>(nested),
	error_single_dimension_standard(),stepper_monte_carlo_uniform(seed_mc),vector_sampler_uniform(seed_vs),adaptive_iterations);
}






