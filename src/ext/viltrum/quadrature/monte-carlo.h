#pragma once

#include <array>
#include <random>
#include "range.h"
#include "integrate.h"
#include "vector-dimensions.h"

template<typename RNG>
class StepperMonteCarloUniform {
    mutable RNG rng;


    // Structure to store the MC sum and number of samples
    // used to compute the average
    template<typename Result>
    struct Samples {
        Result sumatory;
        unsigned long counter;
        Samples() : sumatory(0),counter(0) { }
    };
public:
    template<typename F, typename Float, std::size_t DIM>
    auto init(const F& f, const Range<Float,DIM>& range) const {
        //First element of tuple is sum, second element is number of samples
        return Samples<decltype(f(range.min()))>();
    }

    template<typename F, typename Float, std::size_t DIM, typename Result>
    void step(const F& f, const Range<Float,DIM>& range, Samples<Result>& samples) const {
    	// For the dimension, generate random values
        std::array<Float,DIM> sample;
	    for (std::size_t i=0;i<DIM;++i) {
		    std::uniform_real_distribution<Float> dis(range.min(i),range.max(i));
		    sample[i] = dis(rng);
	    }
        // Then call the integrator
        // in case of perPixel, is to do the MC
        samples.sumatory += range.volume()*f(sample);
        ++samples.counter;
    }

    template<typename F, typename Float, std::size_t DIM, typename Result>
    Result integral(const F& f, const Range<Float,DIM>& range, const Samples<Result>& samples) const {
        // Compute the average MC
        return (samples.counter==0)?decltype(samples.sumatory)(0):(samples.sumatory/Float(samples.counter));
    }

    StepperMonteCarloUniform(RNG&& r) :
        rng(std::forward<RNG>(r)) { }
};

template<typename RNG>
class StepperBinsMonteCarloUniform {
    mutable RNG rng;

    template<typename Result,std::size_t DIMBINS, typename Float, std::size_t DIM>
    struct Samples {
        vector_dimensions<Result,DIMBINS> summatory;
        Range<Float,DIM> range;
        unsigned long counter;
        Samples(std::array<std::size_t,DIMBINS> resolution, const Range<Float,DIM>& range) : summatory(resolution,Result(0)),range(range),counter(0) { }
    };
public:
    template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
    auto init(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range) const {
        using Result = decltype(f(range.min()));
        return Samples<Result,DIMBINS,Float,DIM>(resolution, range);
    }

    //This stepper saves the global range so we can step with a local smaller range and it will still work
    template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename Result>
    void step(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, Samples<Result,DIMBINS,Float,DIM>& samples) const {
    	std::array<Float,DIM> sample;
	    for (std::size_t i=0;i<DIM;++i) {
		    std::uniform_real_distribution<Float> dis(range.min(i),range.max(i));
		    sample[i] = dis(rng);
	    }

        // Does it happens that the sample is outside?
        if (samples.range.is_inside(sample)) {
            // ???? Does DIMBINS is 2 for images?
            // This will make sense as the sample is generate uniformly
            // if the resolution and then f is query.
            std::array<std::size_t,DIMBINS> pos;
            for (std::size_t i=0;i<DIMBINS;++i) {
                pos[i] = std::size_t(resolution[i]*(sample[i] - samples.range.min(i))/(samples.range.max(i) - samples.range.min(i)));
            }
            samples.summatory[pos] += range.volume()*f(sample);
        }
        ++samples.counter;
    }

    template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename Result>
    void integral(Bins& bins, const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, const Samples<Result,DIMBINS,Float,DIM>& samples) const {
        // What is this condition for? samples.counter == 0 -> no sample generated or something like that?
        if (samples.counter == 0) {
            for (auto pos : multidimensional_range(resolution))
                bins(pos) = samples.summatory[pos];
        } else {
            for (auto pos : multidimensional_range(resolution))
                bins(pos) = samples.summatory[pos]*double(samples.summatory.size())/double(samples.counter);
        }
    }

    StepperBinsMonteCarloUniform(RNG&& r) : rng(std::forward<RNG>(r)) { }
};


template<typename RNG>
auto stepper_monte_carlo_uniform(RNG&& rng) {
    return StepperMonteCarloUniform<RNG>(std::forward<RNG>(rng));
}

auto stepper_monte_carlo_uniform(std::size_t seed = std::random_device()()) {
    return stepper_monte_carlo_uniform(std::mt19937_64(seed));
}

template<typename RNG>
auto stepper_bins_monte_carlo_uniform(RNG&& rng) {
    return StepperBinsMonteCarloUniform<RNG>(std::forward<RNG>(rng));
}

auto stepper_bins_monte_carlo_uniform(std::size_t seed = std::random_device()()) {
    return stepper_bins_monte_carlo_uniform(std::mt19937_64(seed));
}


template<typename RNG>
auto integrator_monte_carlo_uniform(RNG&& rng, unsigned long samples) {
    return integrator_stepper(stepper_monte_carlo_uniform(std::forward<RNG>(rng)),samples);
}

auto integrator_monte_carlo_uniform(unsigned long samples, std::size_t seed = std::random_device()()) {
    return integrator_stepper(stepper_monte_carlo_uniform(seed),samples);
}


