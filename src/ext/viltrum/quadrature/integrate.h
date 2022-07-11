#pragma once

#include <array>
#include <tuple>
#include "../multiarray/multiarray.h"
#include "../multiarray/fill.h"
#include "../multiarray/fold.h"
#include "../multiarray/io.h"
#include "../multiarray/split.h"
#include "region.h"
#include "error.h"
#include "rules.h"
#include "nested.h"
#include "range.h"
#include <cmath>
#include <algorithm>

template<typename Q>
class IntegratorQuadrature {
    Q quadrature;

public:
//    IntegratorQuadrature(const Q& q) : quadrature(q) { }
    IntegratorQuadrature(Q&& q) : quadrature(std::forward<Q>(q)) { }

    template<typename F, typename Float, std::size_t DIM>
    auto integrate(const F& f, const Range<Float,DIM>& range) const {
        return region(f,quadrature, range.min(), range.max()).integral();
    }
};

template<typename Q>
auto integrator_quadrature(Q&& q) {
    return IntegratorQuadrature<Q>(std::forward<Q>(q));
}

/**
 * Error returns a tuple [err, dim] where err is floating point and dim is std::size_t: it checks the maximum error on each dimension 
 * from a region R
 */
template<typename N, typename Error> 
class IntegratorAdaptiveTolerance {
    N nested;
    Error error;
    double tolerance; //Tolerance is a double but it is only used for comparison purposes so we do not need to specify precision here

 	template<typename F, typename R>
	typename R::value_type integrate_region(const F& f, const R& r) const {
		auto [err,dim] = error(r);
		if (err < tolerance) return r.integral();
		else {	
			auto subregions = r.split(f, dim);
			typename R::value_type sol = integrate_region(f,subregions.front());
			for (auto it = subregions.begin()+1; it != subregions.end(); ++it)
				sol += integrate_region(f,*it);
			return sol;
		}
	}
   
public:
    template<typename F, typename Float, std::size_t DIM>
    auto integrate(const F& f, const Range<Float,DIM>& range) const {
        return integrate_region(f,region(f,nested, range.min(), range.max()));
    }

    IntegratorAdaptiveTolerance(N&& n, Error&& e, double t) :
        nested(std::forward<N>(n)),error(std::forward<Error>(e)), tolerance(t) { }
};

template<typename N, typename Error>
auto integrator_adaptive_tolerance(N&& nested, Error&& error, double tolerance) {
    return IntegratorAdaptiveTolerance<std::decay_t<N>,std::decay_t<Error>>(std::forward<N>(nested),std::forward<Error>(error),tolerance);
}


template<typename N>
auto integrator_adaptive_tolerance(N&& nested, double tolerance) {
    return integrator_adaptive_tolerance(std::forward<N>(nested), error_single_dimension_standard(), tolerance);
}

template<typename Stepper>
class IntegratorStepper {
    Stepper stepper;
    unsigned long iterations;
public:
    template<typename F, typename Float, std::size_t DIM>
    auto integrate(const F& f, const Range<Float,DIM>& range) const {
        auto data = stepper.init(f,range);
        for (unsigned long i = 0; i<iterations;++i) stepper.step(f,range,data);
        return stepper.integral(f,range,data);
    }

    IntegratorStepper(Stepper&& s, unsigned long i) :
        stepper(std::forward<Stepper>(s)), iterations(i) { }
    IntegratorStepper(const Stepper& s, unsigned long i) :
        stepper(s), iterations(i) { }
	
};

template<typename Stepper>
auto integrator_stepper(Stepper&& stepper, unsigned long iterations) {
    return IntegratorStepper<std::decay_t<Stepper>>(std::forward<Stepper>(stepper), iterations);
}

// This is the adaptive stepper
template<typename N, typename Error>
class StepperAdaptive {
    N nested;
    Error error;
public:
    template<typename R>
    static bool compare_single(const R& a, const R& b) {
	    return std::get<0>(a.extra()) < std::get<0>(b.extra());
    }

public:

    // Init: Create one big region
    template<typename F, typename Float, std::size_t DIM>
    auto init(const F& f, const Range<Float,DIM>& range) const {
        // f, nested and the range values
        auto r = region(f,nested,range.min(),range.max());
        auto errdim = error(r);

        // Again the heap is not store on the structure but giving out 
        // by the template.... why?
        // Store the region and extra information. 
        std::vector<ExtendedRegion<decltype(r),decltype(errdim)> > heap;
        heap.emplace_back(r,errdim); // Ok, strange writting but maybe implicity call the extended region constructor
        return heap;
    }

    template<typename F, typename Float, std::size_t DIM, typename R>
    void step(const F& f, const Range<Float,DIM>& range, std::vector<R>& heap) const {
    	// ExtendedRegion: with error information
        auto r = heap.front(); // Give the "Extended Region" (Region + additional info)
	    auto subregions = r.split(f,std::get<1>(r.extra())); 
	    std::pop_heap(heap.begin(),heap.end(),StepperAdaptive<N,Error>::compare_single<R>);  //????
        
        heap.pop_back();
	    for (auto sr : subregions) {
		    auto errdim = error(sr);
		    heap.emplace_back(sr,errdim); 
		    std::push_heap(heap.begin(), heap.end(), StepperAdaptive<N,Error>::compare_single<R>);
	    }
    }

    template<typename F, typename Float, std::size_t DIM, typename R>
    auto integral(const F& f, const Range<Float,DIM>& range, const std::vector<R>& heap) const {
        auto sol = heap.front().integral();
        for (auto it = heap.begin()+1; it != heap.end(); ++it)
            sol += (*it).integral();
        return sol;
    }

    StepperAdaptive(N&& n, Error&& e) :
        nested(std::forward<N>(n)), error(std::forward<Error>(e)) { }
};

template<typename N, typename Error>
auto stepper_adaptive(N&& nested, Error&& error) {
    return StepperAdaptive<N,Error>(std::forward<N>(nested),std::forward<Error>(error));
}

template<typename N>
auto stepper_adaptive(N&& nested) {
    return stepper_adaptive(std::forward<N>(nested), error_single_dimension_standard());
}

template<typename N, typename Error>
auto integrator_adaptive_iterations(N&& nested, Error&& error, unsigned long iterations) {
    return integrator_stepper(stepper_adaptive(nested,error),iterations);
}

template<typename N>
auto integrator_adaptive_iterations(N&& nested, unsigned long iterations) {
    return integrator_stepper(stepper_adaptive(nested),iterations);
}

template<typename Integrator, typename F, typename Float, std::size_t DIM>
auto integrate(const Integrator& integrator, const F& function, const Range<Float,DIM>& range) {
    return integrator.integrate(function,range);
}

