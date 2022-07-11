#pragma once

#include "integrate-bins-adaptive.h"
#include "monte-carlo.h"
#include "sample-vector.h"

template<typename Nested, typename Error, typename ResidualStepper, typename VectorSampler>
class StepperAdaptiveControlVariates {
	StepperAdaptive<Nested,Error> cv_stepper;
	ResidualStepper residual_stepper;
	VectorSampler vector_sampler;
	unsigned long adaptive_iterations;
    
	template<typename R,typename ResData,typename Sampler>
    struct Data {
		std::vector<R> regions;
		ResData residual_data;
		Sampler vector_sampler;
		unsigned long cv_iterations;
        Data(std::vector<R>&& rs, ResData&& rd, Sampler&& vs) : 
			regions(std::forward<std::vector<R>>(rs)),
			residual_data(std::forward<ResData>(rd)),
			vector_sampler(std::forward<Sampler>(vs)),
			cv_iterations(0) { }
    };
public:
	template<typename F, typename Float, std::size_t DIM>
    auto init(const F& f, const Range<Float,DIM>& range) const {
		auto regions = cv_stepper.init(f,range);
        //residual_stepper.init should not do any calculation at all (should be MC)
        // return Data(std::move(regions),
		// 			residual_stepper.init(f, range),
		// 			vector_sampler(regions));
		
		auto init = residual_stepper.init(f, range);

		using VECTOR_TYPE = typename decltype(regions)::value_type;

		return Data<VECTOR_TYPE, 
					decltype(init), 
					decltype(vector_sampler(regions))>
					(std::move(regions),
					std::move(init),
					vector_sampler(regions));
    }
	
	template<typename F, typename Float, std::size_t DIM, typename R,typename ResData,typename Sampler>
    void step(const F& f, const Range<Float,DIM>& range, Data<R,ResData,Sampler>& data) const
	{
		if (data.cv_iterations<adaptive_iterations) {
			cv_stepper.step(f,range,data.regions);
			++data.cv_iterations;
		} else {
			if (data.cv_iterations == adaptive_iterations) {
				data.vector_sampler = vector_sampler(data.regions);
				++data.cv_iterations;
			}
			auto [index, probability] = data.vector_sampler.sample();
			const R& chosen_region = data.regions[index];
			residual_stepper.step([&] (const std::array<Float,DIM>& x)
		      { return (f(x) - chosen_region.approximation_at(x))/probability; },
			  chosen_region.range(), data.residual_data);
		}
    }
	
	template<typename F, typename Float, std::size_t DIM, typename R,typename ResData,typename Sampler>
    auto integral(const F& f, const Range<Float,DIM>& range, const Data<R,ResData,Sampler>& data) const {
        return cv_stepper.integral(f,range,data.regions) +
				residual_stepper.integral(f,range,data.residual_data);
    }
	
	StepperAdaptiveControlVariates(
		Nested&& nested, Error&& error, ResidualStepper&& rs, VectorSampler&& vs, unsigned long ai) :
			cv_stepper(std::forward<Nested>(nested), std::forward<Error>(error)),
			residual_stepper(std::forward<ResidualStepper>(rs)),
			vector_sampler(std::forward<VectorSampler>(vs)),
			adaptive_iterations(ai) { }
};

template<typename Nested, typename Error, typename ResidualStepper, typename VectorSampler>
auto stepper_adaptive_control_variates(Nested&& nested, Error&& error, ResidualStepper&& residual_stepper, VectorSampler&& vector_sampler, unsigned long adaptive_iterations) {
	return StepperAdaptiveControlVariates<std::decay_t<Nested>,std::decay_t<Error>,std::decay_t<ResidualStepper>,std::decay_t<VectorSampler>>(
		std::forward<Nested>(nested),std::forward<Error>(error),std::forward<ResidualStepper>(residual_stepper),std::forward<VectorSampler>(vector_sampler),adaptive_iterations);
}

template<typename Nested>
auto stepper_adaptive_control_variates(Nested&& nested, unsigned long adaptive_iterations, std::size_t seed_mc = std::random_device()(), std::size_t seed_vs = std::random_device()()) {
	return stepper_adaptive_control_variates(std::forward<Nested>(nested),
	error_single_dimension_standard(),stepper_monte_carlo_uniform(seed_mc),vector_sampler_uniform(seed_vs),adaptive_iterations);
}


template<typename Nested, typename Error, typename ResidualStepper, typename VectorSampler>
class StepperBinsAdaptiveControlVariates {
	StepperBinsAdaptive<Nested,Error> cv_stepper;
	ResidualStepper residual_stepper;
	VectorSampler vector_sampler;
	unsigned long adaptive_iterations;
    
	template<typename R,typename ResData,typename Sampler>
    struct Data {
		std::vector<R> regions;
		ResData residual_data;
		Sampler vector_sampler;
		unsigned long cv_iterations;
        Data(std::vector<R>&& rs, ResData&& rd, Sampler&& vs) : 
			regions(std::forward<std::vector<R>>(rs)),
			residual_data(std::forward<ResData>(rd)),
			vector_sampler(std::forward<Sampler>(vs)),
			cv_iterations(0) { }
    };
public:
	template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
    auto init(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range) const {
		auto regions = cv_stepper.init(resolution,f,range);
        //residual_stepper.init should not do any calculation at all (should be MC)
        // return Data(std::move(regions),
		// 			residual_stepper.init(resolution,f, range),
		// 			vector_sampler(regions));

		auto init = residual_stepper.init(resolution,f, range);

		using VECTOR_TYPE = typename decltype(regions)::value_type;

		return Data<VECTOR_TYPE, 
					decltype(init), 
					decltype(vector_sampler(regions))>
					(std::move(regions),
					std::move(init),
					vector_sampler(regions));
    }
	
	template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename R,typename ResData,typename Sampler>
    void step(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, Data<R,ResData,Sampler>& data) const
	{
		if (data.cv_iterations<adaptive_iterations) {
			cv_stepper.step(resolution,f,range,data.regions);
			++data.cv_iterations;
		} else {
			if (data.cv_iterations == adaptive_iterations) {
				data.vector_sampler = vector_sampler(data.regions);
				++data.cv_iterations;
			}
			auto [index, probability] = data.vector_sampler.sample();
			const R& chosen_region = data.regions[index];
			residual_stepper.step(resolution,[&] (const std::array<Float,DIM>& x)
		      { return (f(x) - chosen_region.approximation_at(x))/probability; },
			  chosen_region.range(), data.residual_data);
		}
    }
	
	template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename R,typename ResData,typename Sampler>
    void integral(Bins& bins, const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, const Data<R,ResData,Sampler>& data) const {
        vector_dimensions<decltype(f(range.min())),DIMBINS> bins_cv(resolution);
        vector_dimensions<decltype(f(range.min())),DIMBINS> bins_residual(resolution);
        cv_stepper.integral(bins_cv,resolution,f,range,data.regions);
        residual_stepper.integral(bins_residual,resolution,f,range,data.residual_data);
        for (auto pos : multidimensional_range(resolution))
            bins(pos) = bins_cv[pos]+bins_residual[pos];
    }
	
	StepperBinsAdaptiveControlVariates(
		Nested&& nested, Error&& error, ResidualStepper&& rs, VectorSampler&& vs, unsigned long ai) :
			cv_stepper(std::forward<Nested>(nested), std::forward<Error>(error)),
			residual_stepper(std::forward<ResidualStepper>(rs)),
			vector_sampler(std::forward<VectorSampler>(vs)),
			adaptive_iterations(ai) { }
};

template<typename Nested, typename Error, typename ResidualStepper, typename VectorSampler>
auto stepper_bins_adaptive_control_variates(Nested&& nested, Error&& error, ResidualStepper&& residual_stepper, VectorSampler&& vector_sampler, unsigned long adaptive_iterations) {
	return StepperBinsAdaptiveControlVariates<std::decay_t<Nested>,std::decay_t<Error>,std::decay_t<ResidualStepper>,std::decay_t<VectorSampler>>(
		std::forward<Nested>(nested),std::forward<Error>(error),std::forward<ResidualStepper>(residual_stepper),std::forward<VectorSampler>(vector_sampler),adaptive_iterations);
}

template<typename Nested>
auto stepper_bins_adaptive_control_variates(Nested&& nested, unsigned long adaptive_iterations, std::size_t seed_mc = std::random_device()(), std::size_t seed_vs = std::random_device()()) {
	return stepper_bins_adaptive_control_variates(std::forward<Nested>(nested),
	error_single_dimension_standard(),stepper_bins_monte_carlo_uniform(seed_mc),vector_sampler_uniform(seed_vs),adaptive_iterations);
}


template<typename Nested, typename Error, typename ResidualStepper, typename VectorSampler>
class StepperBinsAdaptiveStratifiedControlVariates {
	StepperBinsAdaptive<Nested,Error> cv_stepper;
	ResidualStepper residual_stepper;
	VectorSampler vector_sampler;
	unsigned long adaptive_iterations;

    template<typename R, typename ResData, typename Float, std::size_t DIM>
    struct BinData {
        ResData residual_data;
        std::vector<const R*> regions;
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
		for (const auto& r : data.regions) {
            std::array<std::size_t,DIMBINS> start_bin, end_bin;
            for (std::size_t i = 0; i<DIMBINS;++i) {
                start_bin[i] = std::max(std::size_t(0),std::size_t((r.range().min(i) - range.min(i))/drange[i]));
                end_bin[i]   = std::min(resolution[i],std::size_t(0.99f + (r.range().max(i) - range.min(i))/drange[i]));
            }

//            if (start_bin == end_bin) data.bin_data[start_bin].regions.push_back(&r); else 
            for (auto pos : multidimensional_range(start_bin, end_bin)) 
                data.bin_data[pos].regions.push_back(&r);
		}


        for (auto pos : multidimensional_range(resolution)) {
			std::array<Float, DIMBINS> submin, submax;
			for (std::size_t i=0;i<DIMBINS;++i) {
				submin[i] = range.min(i)+pos[i]*drange[i];
				submax[i] = range.min(i)+(pos[i]+1)*drange[i];
			}
			Range<Float, DIMBINS> pixel_range(submin,submax);
			data.bin_data[pos].sampler = vector_sampler(data.bin_data[pos].regions);
			data.bin_data[pos].residual_data = residual_stepper.init(f,pixel_range.intersection_large(range));
        }
        return data;
    }
	
	template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename R, typename ResData>
    void step(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, Data<R,Float,DIM,DIMBINS,ResData>& data) const
	{ 
		std::array<Float,DIMBINS> drange;
        for (std::size_t i=0;i<DIMBINS;++i) drange[i] = (range.max(i) - range.min(i))/Float(resolution[i]);
		for (auto pos : multidimensional_range(resolution)) {
			auto [index, probability] = data.bin_data[pos].sampler.sample();
			const R* chosen_region = data.bin_data[pos].regions[index];
			std::array<Float, DIMBINS> submin, submax;
            for (std::size_t i=0;i<DIMBINS;++i) {
                submin[i] = range.min(i)+pos[i]*drange[i];
                submax[i] = range.min(i)+(pos[i]+1)*drange[i];
            }
			Range<Float, DIMBINS> pixel_range(submin,submax);
			residual_stepper.step([&] (const std::array<Float,DIM>& x) -> decltype(f(x)) { return (f(x) - chosen_region->approximation_at(x))/probability; },
			    pixel_range.intersection_large(chosen_region->range()), data.bin_data[pos].residual_data);
		}
    }
	
	template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename R,typename ResData>
    void integral(Bins& bins, const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, const Data<R,Float,DIM,DIMBINS,ResData>& data) const    {
        cv_stepper.integral(bins,resolution,f,range,data.regions);
        for (auto pos : multidimensional_range(resolution)) 
            bins(pos) += double(data.bin_data.size())*residual_stepper.integral(f,range,data.bin_data[pos].residual_data);
    }
	
	StepperBinsAdaptiveStratifiedControlVariates(
		Nested&& nested, Error&& error, ResidualStepper&& rs, VectorSampler&& vs, unsigned long ai) :
			cv_stepper(std::forward<Nested>(nested), std::forward<Error>(error)),
			residual_stepper(std::forward<ResidualStepper>(rs)),
			vector_sampler(std::forward<VectorSampler>(vs)),
			adaptive_iterations(ai) { }
};

template<typename Nested, typename Error, typename ResidualStepper, typename VectorSampler>
auto stepper_bins_adaptive_stratified_control_variates(Nested&& nested, Error&& error, ResidualStepper&& residual_stepper, VectorSampler&& vector_sampler, unsigned long adaptive_iterations) {
	return StepperBinsAdaptiveStratifiedControlVariates<std::decay_t<Nested>,std::decay_t<Error>,std::decay_t<ResidualStepper>,std::decay_t<VectorSampler>>(
		std::forward<Nested>(nested),std::forward<Error>(error),std::forward<ResidualStepper>(residual_stepper),std::forward<VectorSampler>(vector_sampler),adaptive_iterations);
}

template<typename Nested, typename Error>
auto stepper_bins_adaptive_stratified_control_variates(Nested&& nested, Error&& error, unsigned long adaptive_iterations, std::size_t seed_mc = std::random_device()(), std::size_t seed_vs = std::random_device()()) {
	return stepper_bins_adaptive_stratified_control_variates(std::forward<Nested>(nested),std::forward<Error>(error),stepper_monte_carlo_uniform(seed_mc),vector_sampler_uniform(seed_vs),adaptive_iterations);
}



template<typename Nested>
auto stepper_bins_adaptive_stratified_control_variates(Nested&& nested, unsigned long adaptive_iterations, std::size_t seed_mc = std::random_device()(), std::size_t seed_vs = std::random_device()()) {
	return stepper_bins_adaptive_stratified_control_variates(std::forward<Nested>(nested),
	error_single_dimension_standard(),stepper_monte_carlo_uniform(seed_mc),vector_sampler_uniform(seed_vs),adaptive_iterations);
}






