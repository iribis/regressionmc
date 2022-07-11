#pragma once

#include <random>

template<typename RNG>
class VectorSamplerUniform {
	mutable RNG rng;
		
	class Sampler {
		mutable RNG rng;
		std::size_t size;
	public:
		//Returns position and probability
		std::tuple<std::size_t,double> sample() {
			std::uniform_int_distribution<std::size_t> choose(0,size-1);
			return std::make_tuple(choose(rng),1.0/double(size));
		}
		
		Sampler(const RNG& r, std::size_t s) : rng(r), size(s) {}
		Sampler() {} //This makes things easier although I don't like it
	};

public:
	template<typename T>
	Sampler operator()(const std::vector<T>& v) const {
		std::uniform_int_distribution<std::size_t> choose(0,10000000);
		return Sampler(RNG(choose(rng)),v.size());
	}

    VectorSamplerUniform(RNG&& r) : rng(std::forward<RNG>(r)) { }
};

template<typename RNG>
auto vector_sampler_uniform(RNG&& rng) {
    return VectorSamplerUniform<RNG>(std::forward<RNG>(rng));
}

auto vector_sampler_uniform(std::size_t seed = std::random_device()()) {
    return vector_sampler_uniform(std::mt19937_64(seed));
}



template <typename ERROR, typename RNG>
class VectorSamplerError {	
	using DISTRIBUTION_TYPE = std::discrete_distribution<std::size_t>;

	ERROR error;
	mutable RNG rng;
	
	class Sampler {
		mutable RNG rng;
		DISTRIBUTION_TYPE distribution;
	public:
		//Returns position and probability
		std::tuple<std::size_t,double> sample() {
			//std::uniform_int_distribution<std::size_t> choose(0,size-1);
			//return std::make_tuple(choose(rng),1.0/double(size));

			std::size_t index = distribution(rng);
			return std::make_tuple(index, distribution.probabilities()[index]);
		}
		
		Sampler(const RNG& r, DISTRIBUTION_TYPE&& dist) : rng(r), distribution(std::forward<DISTRIBUTION_TYPE>(dist)) {}
		Sampler() {} //This makes things easier although I don't like it
	};

public:
	template<typename T>
	Sampler operator()(const std::vector<T>& v) const {
		std::vector<double> weights;
		weights.resize(v.size());
		for(std::size_t i=0; i<v.size(); i++) {
			weights[i] = std::get<0>(error(*(v[i])));
		}
		DISTRIBUTION_TYPE dist(weights.begin(), weights.end());

		std::uniform_int_distribution<std::size_t> choose(0,10000000);
		return Sampler(RNG(choose(rng)),std::forward<DISTRIBUTION_TYPE>(dist));
	}

    VectorSamplerError(ERROR&& e, RNG&& r) : error(std::forward<ERROR>(e)), rng(std::forward<RNG>(r)) { }
};

template<typename ERROR>
auto vector_sampler_error(ERROR&& error, std::size_t seed = std::random_device()()) {
	using RNG = decltype(std::mt19937_64(seed));
    return VectorSamplerError<ERROR, RNG>(std::forward<ERROR>(error),std::mt19937_64(seed));
}

class VectorSamplerstratified {	
	
	class Sampler {
		std::size_t index;
		std::size_t max_values;
	public:
		//Returns position and probability
		std::tuple<std::size_t,double> sample() {
			std::size_t res = index;
			index++;
			if (index >= max_values) index = 0;
			//return std::make_tuple(res, 1.0/static_cast<float>(max_values));
			return std::make_tuple(res, 1.0);
		}
		
		Sampler(std::size_t max_number) : max_values(max_number), index(0) {}
		Sampler() {} //This makes things easier although I don't like it
	};

public:
	template<typename T>
	Sampler operator()(const std::vector<T>& v) const {
		return Sampler(v.size());
	}

    //VectorSamplerstratified() { }
};

auto vector_sampler_stratified() {
    return VectorSamplerstratified();
}