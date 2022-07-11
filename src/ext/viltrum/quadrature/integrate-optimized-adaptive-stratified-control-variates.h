#pragma once

// Their include
// - Quadrature rules and the subdivision behavior
#include "integrate.h"
// - Seems a tools to have smart iterators
// 	for vector with different ranges
#include "multidimensional-range.h"

// STL
#include <vector>
#include <random>
#include <vector>
#include <array>
#include <type_traits>

//#define MEASURE_TIME
#ifdef MEASURE_TIME
#include <chrono>
//using seconds = std::chrono::seconds;
using seconds = std::chrono::milliseconds;
using nano = std::chrono::microseconds;
#endif

// This is the region generator that it is used.
// Error is just the function to compute the error
// Nested: I am not sure
template <typename Nested, typename Error>
class RegionGenerator
{
	// Seems that all interesting code are inside stepperAdaptive
	StepperAdaptive<Nested, Error> stepper;
	unsigned long adaptive_iterations;

public:
	RegionGenerator(Nested &&nested, Error &&error, unsigned long ai) : stepper(std::forward<Nested>(nested), std::forward<Error>(error)),
																		adaptive_iterations(ai) {}

	template <typename F, typename Float, std::size_t DIM>
	auto compute_regions(const F &f, const Range<Float, DIM> &range) const
	{
		// Maybe the number of iteration is based on the number of samples?
		// Initialize the only big region
		auto regions = stepper.init(f, range);
		
		// Adaptively subdivide the regions
		for (unsigned long i = 0; i < adaptive_iterations; ++i)
		{
			stepper.step(f, range, regions);
		}
		return regions;
	}
};

// Constructor for the region generator.
template <typename Nested, typename Error>
auto region_generator(Nested &&nested, Error &&error, unsigned long adaptive_iterations)
{
	return RegionGenerator<std::decay_t<Nested>, std::decay_t<Error>>(
		std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations);
}

template <typename Float, std::size_t DIM, std::size_t DIMBINS, typename R>
auto pixels_in_region(const R &r, const std::array<std::size_t, DIMBINS> &bin_resolution, const Range<Float, DIM> &range)
{
	std::array<std::size_t, DIMBINS> start_bin, end_bin;
	for (std::size_t i = 0; i < DIMBINS; ++i)
	{
		start_bin[i] = std::max(std::size_t(0), std::size_t(Float(bin_resolution[i]) * (r.range().min(i) - range.min(i)) / (range.max(i) - range.min(i))));
		end_bin[i] = std::max(start_bin[i] + 1, std::min(bin_resolution[i], std::size_t(0.99f +
																						(Float(bin_resolution[i]) * (r.range().max(i) - range.min(i)) / (range.max(i) - range.min(i))))));
	}
	return multidimensional_range(start_bin, end_bin);
}

class AlphaOptimized
{
	double alpha_min, alpha_max;

	template <typename T>
	T alpha(const T &cov, const T &var, typename std::enable_if<std::is_floating_point<T>::value>::type * = 0) const
	{
		// Compute alpha
		return (var == T(0)) ? ((cov > T(0)) ? T(0) : T(alpha_max)) : std::max(T(alpha_min), std::min(T(alpha_max), (cov / var)));
	}

	template <typename T>
	T alpha(const T &cov, const T &var, typename std::enable_if<!std::is_floating_point<T>::value>::type * = 0) const
	{
		T res;
		for (decltype(res.size()) i = 0; i < res.size(); i++)
		{
			res[i] = alpha(cov[i], var[i]);
		}
		return res;
	}

public:
	AlphaOptimized(double amin = -1.0, double amax = 1.0) : alpha_min(amin), alpha_max(amax) {}

	// Code that takes the samples
	// and compute alpha
	template <typename T>
	T alpha(const std::vector<std::tuple<T, T>> &samples) const
	{
		if (samples.size() <= 1)
			return T(1);

		// Compute f and app mean
		T mean_f, mean_app;
		std::tie(mean_f, mean_app) = samples[0];
		for (std::size_t s = 1; s < samples.size(); ++s)
		{
			mean_f += std::get<0>(samples[s]);
			mean_app += std::get<1>(samples[s]);
		}
		// Normalize
		mean_f /= double(samples.size());
		mean_app /= double(samples.size());

		// Compute covariance  (var of the two components)
		// and the variance of residual
		T cov = (std::get<0>(samples[0]) - mean_f) * (std::get<1>(samples[0]) - mean_app);
		T var_app = (std::get<1>(samples[0]) - mean_app) * (std::get<1>(samples[0]) - mean_app);
		for (std::size_t s = 1; s < samples.size(); ++s)
		{
			cov += (std::get<0>(samples[s]) - mean_f) * (std::get<1>(samples[s]) - mean_app);
			var_app += (std::get<1>(samples[s]) - mean_app) * (std::get<1>(samples[s]) - mean_app);
		}
		//		cov/=double(samples.size()); var_app/=double(samples.size());
		//		^^ unnecesary division as it gets simplified afterwards

		// Get the alpha value
		return alpha(cov, var_app);
	}
};

// Alpha constant = 1.0
class AlphaConstant
{
	double value;

public:
	AlphaConstant(double a = 1.0) : value(a) {}

	template <typename T>
	T alpha(const std::vector<std::tuple<T, T>> &samples) const
	{
		return T(value);
	}
};

// Observation: Seems quite old code here
// because they uses functor type of approach
class FunctionSampler
{
public:
	template <typename F, typename Float, std::size_t DIM, typename RNG>
	auto sample(const F &f, const Range<Float, DIM> &range, RNG &rng) const
	{
		std::array<Float, DIM> sample;
		for (std::size_t i = 0; i < DIM; ++i)
		{
			std::uniform_real_distribution<Float> dis(range.min(i), range.max(i));
			sample[i] = dis(rng);
		}
		// Gives both the function values and the coordinates
		return std::make_tuple(f(sample), sample);
	}
};

// Compute the range of a given pixel
// pixel: The pixel position
// bin_resolution: "how many pixels"
// range: the range
// The code will offset the min max and compute the min/max range
template <typename Float, std::size_t DIM, std::size_t DIMBINS>
Range<Float, DIMBINS> range_of_pixel(const std::array<std::size_t, DIMBINS> &pixel,
									 const std::array<std::size_t, DIMBINS> &bin_resolution, 
									 const Range<Float, DIM> &range)
{
	// For the different dimensions
	std::array<Float, DIMBINS> submin, submax;
	for (std::size_t i = 0; i < DIMBINS; ++i)
	{
		submin[i] = range.min(i) + pixel[i] * (range.max(i) - range.min(i)) / Float(bin_resolution[i]);
		submax[i] = range.min(i) + (pixel[i] + 1) * (range.max(i) - range.min(i)) / Float(bin_resolution[i]);
	}
	return Range<Float, DIMBINS>(submin, submax);
}

template <typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
class IntegratorStratifiedAllControlVariates
{
	RegionGenerator region_generator;
	AlphaCalculator alpha_calculator;
	Sampler sampler;
	mutable RNG rng;
	unsigned long spp;

public:
	typedef void is_integrator_tag;

	template <typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
	void integrate(Bins &bins, const std::array<std::size_t, DIMBINS> &bin_resolution, const F &f, const Range<Float, DIM> &range) const
	{
		auto regions = region_generator.compute_regions(f, range);
		using value_type = decltype(f(range.min()));
		using R = typename decltype(regions)::value_type;
		vector_dimensions<std::vector<const R *>, DIMBINS> regions_per_pixel(bin_resolution);
		for (const auto &r : regions)
			for (auto pixel : pixels_in_region(r, bin_resolution, range))
				regions_per_pixel[pixel].push_back(&r);
		for (auto pixel : multidimensional_range(bin_resolution))
		{ // Per pixel
			bins(pixel) = value_type(0);
			auto pixel_range = range_of_pixel(pixel, bin_resolution, range);
			const auto &regions_here = regions_per_pixel[pixel];
			std::size_t samples_per_region = spp / regions_here.size();
			std::size_t samples_per_region_rest = spp % regions_here.size();
			std::uniform_int_distribution<std::size_t> sr(std::size_t(0), regions_here.size() - 1);
			std::size_t sampled_region = sr(rng);
			for (std::size_t r = 0; r < regions_here.size(); ++r)
			{ // Per region inside the pixel
				std::size_t nsamples = samples_per_region;
				if (((r - sampled_region) % (regions_here.size())) < samples_per_region_rest)
					nsamples += 1;
				auto local_range = pixel_range.intersection_large(regions_here[r]->range());
				if (nsamples == 0) // Case quad?
					bins(pixel) += double(regions_per_pixel.size()) * regions_here[r]->integral_subrange(local_range);
				else
				{
					// 1/M_regions * 1/M_pixels? (and voluem for the integration proposes)
					std::vector<std::tuple<value_type, value_type>> samples(nsamples);
					double factor = local_range.volume() * double(regions_per_pixel.size()) * double(regions_here.size());
					//  ^^ MC probability      ^^ Global size (res-constant)     ^^ Region probability

					for (auto &s : samples)
					{
						auto [value, sample] = sampler.sample(f, local_range, rng);
						s = std::make_tuple(factor * value, factor * regions_here[r]->approximation_at(sample));
					}
					auto a = alpha_calculator.alpha(samples);
					value_type residual = (std::get<0>(samples[0]) - a * std::get<1>(samples[0]));
					for (std::size_t s = 1; s < nsamples; ++s)
						residual += (std::get<0>(samples[s]) - a * std::get<1>(samples[s]));

					//We are multiplying all the samples by the number of regions so we do this
					bins(pixel) += (residual / double(spp)); //... instead of this -> (residual/double(nsamples))
					bins(pixel) += double(regions_per_pixel.size()) * a * regions_here[r]->integral_subrange(local_range);
					//If we covered each region independently we would not multiply by the number of regions
				}
			}
		}
	}

	IntegratorStratifiedAllControlVariates(RegionGenerator &&region_generator,
										   AlphaCalculator &&alpha_calculator, Sampler &&sampler, RNG &&rng, unsigned long spp) : region_generator(std::forward<RegionGenerator>(region_generator)),
																																  alpha_calculator(std::forward<AlphaCalculator>(alpha_calculator)),
																																  sampler(std::forward<Sampler>(sampler)),
																																  rng(std::forward<RNG>(rng)), spp(spp) {}
};

// This is the integrator that it is used in 4D
// My guess is that the samplers and other template argument 
// it is for "speed". Not sure how much they gain from this type of optimization
template <typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
class IntegratorStratifiedPixelControlVariates
{
	RegionGenerator region_generator; //< To generate the different regions (adaptively subdivide)
	AlphaCalculator alpha_calculator; //< Opt or constant

	Sampler sampler;	//< TODO: Why there is a difference between sampler and rng here???
	mutable RNG rng;	// Mutable certainly due to const somehere
	unsigned long spp;	//< The SPP (including CV?) -- I don't think so

public:
	typedef void is_integrator_tag;

	template <typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
	void integrate(Bins &bins, const std::array<std::size_t, DIMBINS> &bin_resolution, const F &f, const Range<Float, DIM> &range) const
	{

		// f: if the warpper used to evaluate the approach
		// This call create N regions
		auto regions = region_generator.compute_regions(f, range);

		// For the different pixel, we will just push the regions that 
		// overlap with the given pixel
		using value_type = decltype(f(range.min()));
		using R = typename decltype(regions)::value_type;
		vector_dimensions<std::vector<const R *>, DIMBINS> regions_per_pixel(bin_resolution);

		// Fill the pixel <-> region vector
		for (const auto &r : regions)
			for (auto pixel : pixels_in_region(r, bin_resolution, range))
				regions_per_pixel[pixel].push_back(&r);

		// This is certainly to go over all the pixels
		// But not sure what is the "bin_resolution" ... OK
		// BIN can be 1D, 2D or 3D (pixel, image, video)
		for (auto pixel : multidimensional_range(bin_resolution))
		{ // Per pixel

			// Ok, this initialize the value to 0 
			// For this pixel
			bins(pixel) = value_type(0);
			
			// Ok certainly an approach to stratify samples per regions
			// But seems more complicated that it should be done
			auto pixel_range = range_of_pixel(pixel, bin_resolution, range);
			const auto &regions_here = regions_per_pixel[pixel]; // All the regions for this pixel

			// Then compute the SPP and how it is allocated per regions
			std::size_t samples_per_region = spp / regions_here.size();
			std::size_t samples_per_region_rest = spp % regions_here.size();
			std::vector<std::tuple<value_type, value_type>> samples;
			samples.reserve(spp);

			//First: stratified distribution of samples (uniformly)
			// Note that it is if samples_per_regions > 1
			// For sample stratification
			for (std::size_t r = 0; r < regions_here.size(); ++r)
			{
				// Do the intersection between the pixel_range and the region range
				auto local_range = pixel_range.intersection_large(regions_here[r]->range());
				
				// This is all the normalization factors
				double factor = local_range.volume() * double(regions_per_pixel.size()) * double(regions_here.size());
				for (std::size_t s = 0; s < samples_per_region; ++s)
				{
					// Sample the intersected region: This also make an evaluation of the function
					auto [value, sample] = sampler.sample(f, local_range, rng);

					// This is the quadrature value, I guess.
					auto approxValue = regions_here[r]->approximation_at(sample);

					// Push the sample and the CV value (for later computation)
					samples.push_back(std::make_tuple(factor * value, factor * approxValue));
				}
			}

			// Randomly samples the rest of the domain. 
			// I think they should have sampled or discretize the decision.
			// This will makes an easier implementation.
			std::uniform_int_distribution<std::size_t> sample_region(std::size_t(0), regions_here.size() - 1);
			//We randomly distribute the rest of samples among all regions
			for (std::size_t i = 0; i < samples_per_region_rest; ++i)
			{
				std::size_t r = sample_region(rng);
				auto local_range = pixel_range.intersection_large(regions_here[r]->range());
				double factor = local_range.volume() * double(regions_per_pixel.size()) * double(regions_here.size());

				auto [value, sample] = sampler.sample(f, local_range, rng);

				auto approxValue = regions_here[r]->approximation_at(sample);
				samples.push_back(std::make_tuple(factor * value, factor * approxValue));
			}

			// Compute alpha and residual
			value_type a(1.0);
			if (spp != 0)
			{
				a = alpha_calculator.alpha(samples);

				// Resuidual control variate.
				value_type residual = (std::get<0>(samples[0]) - a * std::get<1>(samples[0]));
				for (std::size_t s = 1; s < spp; ++s)
					residual += (std::get<0>(samples[s]) - a * std::get<1>(samples[s]));
				
				// Save the results inside the pixel (certainly that)
				bins(pixel) += (residual / double(spp));
			}

			// Control variate part (again, pdf -> regions per pixel, and then the integral subrange)
			for (auto r : regions_here)
				bins(pixel) += double(regions_per_pixel.size()) * a * r->integral_subrange(pixel_range.intersection_large(r->range()));
		}
	}

	IntegratorStratifiedPixelControlVariates(RegionGenerator &&region_generator,
											 AlphaCalculator &&alpha_calculator, Sampler &&sampler, RNG &&rng, unsigned long spp) : region_generator(std::forward<RegionGenerator>(region_generator)),
																																	alpha_calculator(std::forward<AlphaCalculator>(alpha_calculator)),
																																	sampler(std::forward<Sampler>(sampler)),
																																	rng(std::forward<RNG>(rng)), spp(spp) {}
};

template <typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
class IntegratorStratifiedRegionControlVariates
{
	RegionGenerator region_generator;
	AlphaCalculator alpha_calculator;
	Sampler sampler;
	mutable RNG rng;
	unsigned long spp;

public:
	typedef void is_integrator_tag;

	template <typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
	void integrate(Bins &bins, const std::array<std::size_t, DIMBINS> &bin_resolution, const F &f, const Range<Float, DIM> &range) const
	{
		auto regions = region_generator.compute_regions(f, range);
		using value_type = decltype(f(range.min()));

		auto all_pixels = multidimensional_range(bin_resolution);

		for (auto pixel : all_pixels)
			bins(pixel) = value_type(0);

		std::size_t samples_per_region = (spp * all_pixels) / regions.size();
		std::size_t samples_per_region_rest = (spp * all_pixels) % regions.size();
		std::uniform_int_distribution<std::size_t> sr(std::size_t(0), regions.size() - 1);
		std::size_t sampled_region = sr(rng);

		//		std::cerr<<"Regions = "<<regions.size()<<" - Samples per region "<<samples_per_region<<" - Rest = "<<samples_per_region_rest<<std::endl;

		std::size_t ri = 0;
		for (const auto &r : regions)
		{
			std::size_t nsamples = samples_per_region +
								   (((((ri++) + sampled_region) % regions.size()) < samples_per_region_rest) ? 1 : 0);
			auto pixels = pixels_in_region(r, bin_resolution, range);

			std::size_t samples_per_pixel = nsamples / pixels;
			std::size_t samples_per_pixel_rest = nsamples % pixels;

			//			std::cerr<<"    Region "<<ri<<" -> "<<nsamples<<" samples, "<<int(pixels)<<" pixels, "<<samples_per_pixel<<" samples per pixel, "<<samples_per_pixel_rest<<" random samples"<<std::endl;

			std::vector<std::tuple<value_type, value_type>> samples;
			samples.reserve(nsamples);
			std::vector<std::array<std::size_t, DIMBINS>> positions;
			positions.reserve(nsamples);

			for (auto pixel : pixels)
			{
				auto pixel_range = range_of_pixel(pixel, bin_resolution, range).intersection_large(r.range());
				double factor = pixel_range.volume() * double(all_pixels) * double(pixels);
				for (std::size_t pass = 0; pass < samples_per_pixel; ++pass)
				{
					auto [value, sample] = sampler.sample(f, pixel_range, rng);
					samples.push_back(std::make_tuple(factor * value, factor * r.approximation_at(sample)));
					positions.push_back(pixel);
				}
			}

			double factor = r.range().volume() * double(all_pixels);
			for (std::size_t i = 0; i < samples_per_pixel_rest; ++i)
			{
				auto [value, sample] = sampler.sample(f, r.range(), rng);
				samples.push_back(std::make_tuple(factor * value, factor * r.approximation_at(sample)));
				std::array<std::size_t, DIMBINS> pixel;
				for (std::size_t i = 0; i < DIMBINS; ++i)
					pixel[i] = std::size_t(bin_resolution[i] * (sample[i] - range.min(i)) / (range.max(i) - range.min(i)));
				positions.push_back(pixel);
			}

			auto a = alpha_calculator.alpha(samples);
			double residual_factor = 1.0 / double(nsamples);

			for (std::size_t s = 0; s < nsamples; ++s)
			{
				bins(positions[s]) += residual_factor * (std::get<0>(samples[s]) - a * std::get<1>(samples[s]));
			}
			for (auto pixel : pixels)
			{
				auto pixel_range = range_of_pixel(pixel, bin_resolution, range).intersection_large(r.range());
				bins(pixel) += (a * double(all_pixels)) * r.integral_subrange(pixel_range);
			}
		}
	}

	IntegratorStratifiedRegionControlVariates(RegionGenerator &&region_generator,
											  AlphaCalculator &&alpha_calculator, Sampler &&sampler, RNG &&rng, unsigned long spp) : region_generator(std::forward<RegionGenerator>(region_generator)),
																																	 alpha_calculator(std::forward<AlphaCalculator>(alpha_calculator)),
																																	 sampler(std::forward<Sampler>(sampler)),
																																	 rng(std::forward<RNG>(rng)), spp(spp) {}
};

template <typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
auto integrator_stratified_all_control_variates(RegionGenerator &&rg,
												AlphaCalculator &&alpha_calculator, Sampler &&sampler, RNG &&rng, unsigned long spp)
{
	return IntegratorStratifiedAllControlVariates<
		std::decay_t<RegionGenerator>, std::decay_t<AlphaCalculator>, std::decay_t<Sampler>, std::decay_t<RNG>>(
		std::forward<RegionGenerator>(rg),
		std::forward<AlphaCalculator>(alpha_calculator),
		std::forward<Sampler>(sampler),
		std::forward<RNG>(rng),
		spp);
}

template <typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
auto integrator_stratified_pixel_control_variates(RegionGenerator &&rg,
												  AlphaCalculator &&alpha_calculator, Sampler &&sampler, RNG &&rng, unsigned long spp)
{
	return IntegratorStratifiedPixelControlVariates<
		std::decay_t<RegionGenerator>, std::decay_t<AlphaCalculator>, std::decay_t<Sampler>, std::decay_t<RNG>>(
		std::forward<RegionGenerator>(rg),
		std::forward<AlphaCalculator>(alpha_calculator),
		std::forward<Sampler>(sampler),
		std::forward<RNG>(rng),
		spp);
}

template <typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
auto integrator_stratified_region_control_variates(RegionGenerator &&rg,
												   AlphaCalculator &&alpha_calculator, Sampler &&sampler, RNG &&rng, unsigned long spp)
{
	return IntegratorStratifiedRegionControlVariates<
		std::decay_t<RegionGenerator>, std::decay_t<AlphaCalculator>, std::decay_t<Sampler>, std::decay_t<RNG>>(
		std::forward<RegionGenerator>(rg),
		std::forward<AlphaCalculator>(alpha_calculator),
		std::forward<Sampler>(sampler),
		std::forward<RNG>(rng),
		spp);
}

template <typename Nested, typename Error, typename RNG>
auto integrator_optimized_adaptive_stratified_control_variates(Nested &&nested, Error &&error,
															   unsigned long adaptive_iterations, unsigned long spp, RNG &&rng)
{

	return integrator_stratified_all_control_variates(region_generator(std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations), AlphaOptimized(), FunctionSampler(), std::forward<RNG>(rng), spp);
}

template <typename Nested, typename Error, typename RNG>
auto integrator_optimized_perpixel_adaptive_stratified_control_variates(Nested &&nested, Error &&error,
																		unsigned long adaptive_iterations, unsigned long spp, RNG &&rng)
{

	return integrator_stratified_pixel_control_variates(region_generator(std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations), AlphaOptimized(), FunctionSampler(), std::forward<RNG>(rng), spp);
}

template <typename Nested, typename Error, typename RNG>
auto integrator_optimized_perregion_adaptive_stratified_control_variates(Nested &&nested, Error &&error,
																		 unsigned long adaptive_iterations, unsigned long spp, RNG &&rng)
{

	return integrator_stratified_region_control_variates(region_generator(std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations), AlphaOptimized(), FunctionSampler(), std::forward<RNG>(rng), spp);
}

template <typename Nested, typename Error, typename RNG>
auto integrator_alpha1_perregion_adaptive_stratified_control_variates(Nested &&nested, Error &&error,
																	  unsigned long adaptive_iterations, unsigned long spp, RNG &&rng)
{

	return integrator_stratified_region_control_variates(region_generator(std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations), AlphaConstant(), FunctionSampler(), std::forward<RNG>(rng), spp);
}

template <typename Nested, typename Error>
auto integrator_optimized_adaptive_stratified_control_variates(Nested &&nested, Error &&error,
															   unsigned long adaptive_iterations, unsigned long spp, std::size_t seed = std::random_device()())
{

	return integrator_optimized_adaptive_stratified_control_variates(
		std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations, spp, std::mt19937_64(seed));
}

template <typename Nested, typename Error>
auto integrator_optimized_perpixel_adaptive_stratified_control_variates(Nested &&nested, Error &&error,
																		unsigned long adaptive_iterations, unsigned long spp, std::size_t seed = std::random_device()())
{

	return integrator_optimized_perpixel_adaptive_stratified_control_variates(
		std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations, spp, std::mt19937_64(seed));
}

template <typename Nested, typename Error>
auto integrator_optimized_perregion_adaptive_stratified_control_variates(Nested &&nested, Error &&error,
																		 unsigned long adaptive_iterations, unsigned long spp, std::size_t seed = std::random_device()())
{

	return integrator_optimized_perregion_adaptive_stratified_control_variates(
		std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations, spp, std::mt19937_64(seed));
}

template <typename Nested, typename Error>
auto integrator_alpha1_perregion_adaptive_stratified_control_variates(Nested &&nested, Error &&error,
																	  unsigned long adaptive_iterations, unsigned long spp, std::size_t seed = std::random_device()())
{

	return integrator_alpha1_perregion_adaptive_stratified_control_variates(
		std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations, spp, std::mt19937_64(seed));
}
