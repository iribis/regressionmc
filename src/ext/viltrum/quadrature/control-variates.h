#pragma once

#include "region.h"
#include "monte-carlo.h"
#include "integrate-bins-stepper.h"

/**
 * A control variate should be evaluable and integrable 
 *      Result operator()(const std::array<Float,DIM>& x) 
 *      Result integral(F function, Range range)
 */

/**
 * CV returns a control variate from the input function
 */
template<typename RS, typename CVG>
class StepperControlVariate {
    RS residual_stepper;
    CVG cv_generator;
    
    template<typename Residual, typename CV>
    struct Data {
        Residual residual;
        CV control_variate;
        Data(Residual&& r, CV&& c) : residual(std::forward<Residual>(r)),control_variate(std::forward<CV>(c)) { }
        Data(const Residual& r, CV&& c) : residual(r),control_variate(std::forward<CV>(c)) { }
        Data(Residual&& r, const CV& c) : residual(std::forward<Residual>(r)),control_variate(c) { }
        Data(const Residual& r, const CV& c) : residual(r),control_variate(c) { }
    };

    template<typename Samples, typename CV>
    static auto data(Samples&& samples, CV&& cv) { return Data<std::decay_t<Samples>,std::decay_t<CV>>(std::forward<Samples>(samples),std::forward<CV>(cv)); }

public:
    template<typename F, typename Float, std::size_t DIM>
    auto init(const F& f, const Range<Float,DIM>& range) const {
        auto control_variate = cv_generator(f,range);
        auto residual = residual_stepper.init([&] (const std::array<Float,DIM>& x) { return f(x) - control_variate(x); }, range);
        return data(residual, control_variate);
    }

    template<typename F, typename Float, std::size_t DIM, typename Residual, typename CV>
    void step(const F& f, const Range<Float,DIM>& range, Data<Residual,CV>& data) const {
        residual_stepper.step([&] (const std::array<Float,DIM>& x) { return f(x) - data.control_variate(x); }, range, data.residual);
    }

    template<typename F, typename Float, std::size_t DIM, typename Residual, typename CV>
    auto integral(const F& f, const Range<Float,DIM>& range, const Data<Residual,CV>& data) const {
        return data.control_variate.integral(f,range) + residual_stepper.integral(f,range,data.residual);
    }

    StepperControlVariate(RS&& r, CVG&& c) :
        residual_stepper(std::forward<RS>(r)), cv_generator(std::forward<CVG>(c)) { }
};

/**
 * A control variate should be evaluable and integrable 
 *      Result operator()(const std::array<Float,DIM>& x) 
 *      Result integral(F function, Range range)
 */

/**
 * CV returns a control variate from the input function
 */
template<typename RS, typename CVG>
class StepperBinsControlVariate {
    RS residual_stepper;
    CVG cv_generator;
    
    template<typename Residual, typename CV>
    struct Data {
        Residual residual;
        CV control_variate;
        Data(Residual&& r, CV&& c) : residual(std::forward<Residual>(r)),control_variate(std::forward<CV>(c)) { }
        Data(const Residual& r, CV&& c) : residual(r),control_variate(std::forward<CV>(c)) { }
        Data(Residual&& r, const CV& c) : residual(std::forward<Residual>(r)),control_variate(c) { }
        Data(const Residual& r, const CV& c) : residual(r),control_variate(c) { }
    };

    template<typename Samples, typename CV>
    static auto data(Samples&& samples, CV&& cv) { return Data<std::decay_t<Samples>,std::decay_t<CV>>(std::forward<Samples>(samples),std::forward<CV>(cv)); }

public:
    template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
    auto init(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range) const {
        auto control_variate = cv_generator(f,range);
        auto residual = residual_stepper.init(resolution, [&] (const std::array<Float,DIM>& x) { return f(x) - control_variate(x); }, range);
        return data(residual, control_variate);
    }

    template<std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename Residual, typename CV>
    void step(const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, Data<Residual,CV>& data) const {
        residual_stepper.step(resolution, [&] (const std::array<Float,DIM>& x) { return f(x) - data.control_variate(x); }, range, data.residual);
    }

    template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM, typename Residual, typename CV>
    void integral(Bins& bins, const std::array<std::size_t,DIMBINS>& resolution, const F& f, const Range<Float,DIM>& range, const Data<Residual,CV>& data) const {
        residual_stepper.integral(bins, resolution, f,range,data.residual);
        std::array<Float,DIMBINS> drange; for (std::size_t i=0;i<DIMBINS;++i) drange[i] = (range.max(i) - range.min(i))/Float(resolution[i]);
        double factor = 1;  for (std::size_t i=0;i<DIMBINS;++i) factor*=resolution[i];
        for (auto pos : multidimensional_range(resolution)) {
            Range<Float,DIM> subrange = range;
            for (std::size_t i=0;i<DIMBINS;++i) 
                subrange = subrange.subrange_dimension(i,range.min(i)+pos[i]*drange[i],range.min(i)+(pos[i]+1)*drange[i]);
            bins(pos) += factor*data.control_variate.integral(f,subrange);
        }

    }

    StepperBinsControlVariate(RS&& r, CVG&& c) :
        residual_stepper(std::forward<RS>(r)), cv_generator(std::forward<CVG>(c)) { }
};


template<typename Rule>
class ControlVariateQuadrature {
    Rule rule;

    template<typename Float, std::size_t DIM, typename VT>
    class Function {
        Region<Float,Rule,DIM,VT> reg;
    public:
        VT operator()(const std::array<Float,DIM>& x) const {
            return reg.approximation_at(x);
        }
        template<typename F, std::size_t DIMSUB>
        VT integral(const F& f, const Range<Float,DIMSUB>& range) const { return reg.integral_subrange(range); }
        Function(Region<Float,Rule,DIM,VT>&& r) : reg(std::forward<Region<Float,Rule,DIM,VT>>(r)) {}
    };

public:
    template<typename F, typename Float, std::size_t DIM>
    auto operator()(const F& f, const Range<Float,DIM>& range) const {
        return Function<Float,DIM,decltype(f(range.min()))>(region(f,rule,range.min(),range.max()));   
    }

    ControlVariateQuadrature(Rule&& r) : rule(std::forward<Rule>(r)) { }
    ControlVariateQuadrature(const Rule& r) : rule(r) { }
};

template<typename Nested, typename Error>
class ControlVariateQuadratureAdaptive {
    StepperAdaptive<Nested,Error> stepper;
    unsigned long adaptive_iterations;
    
    template<typename R>
    class Function {
        std::vector<R> regions;
    public:
        template<typename Float>
        typename R::value_type operator()(const std::array<Float,R::dimensions>& x) const {
            for (const auto& r : regions) if (r.range().is_inside(x)) return r.approximation_at(x);
            //Default behavior: extrapolate first region if it is outside all of them
            return regions.front().approximation_at(x);
        }
        template<typename F, typename Float, std::size_t DIM>
        typename R::value_type integral(const F& f, const Range<Float,DIM>& range) const {
            auto sol = regions.front().integral_subrange(range.intersection(regions.front().range()));
            for (auto it = regions.begin()+1; it != regions.end(); ++it)
                sol += (*it).integral_subrange(range.intersection((*it).range()));
            return sol;
        }

        const std::vector<R>& get_regions() const { return regions; }

        Function(std::vector<R>&& rs) : regions(std::forward<std::vector<R>>(rs)) { }
    };

    template<typename R>
    Function<R> function(std::vector<R>&& regions) const { return Function<R>(std::forward<std::vector<R>>(regions)); }
public:
    template<typename F, typename Float, std::size_t DIM>
    auto operator()(const F& f, const Range<Float,DIM>& range) const {
        auto regions = stepper.init(f,range);
        for (unsigned long i = 0; i<adaptive_iterations;++i) stepper.step(f,range,regions);
        return function(std::move(regions));
    }

    ControlVariateQuadratureAdaptive(Nested&& nested, Error&& error, unsigned long ai) :
        stepper(std::forward<Nested>(nested), std::forward<Error>(error)), adaptive_iterations(ai) { }
};

template<typename Rule>
auto control_variate_quadrature(Rule&& rule) { return ControlVariateQuadrature<std::decay_t<Rule>>(std::forward<Rule>(rule)); }

template<typename Nested, typename Error>
auto control_variate_quadrature_adaptive(Nested&& nested, Error&& error, unsigned long adaptive_iterations) {
    return ControlVariateQuadratureAdaptive<std::decay_t<Nested>,std::decay_t<Error>>(std::forward<Nested>(nested),std::forward<Error>(error), adaptive_iterations);
}

template<typename Nested>
auto control_variate_quadrature_adaptive(Nested&& nested, unsigned long adaptive_iterations) {
    return control_variate_quadrature_adaptive(std::forward<Nested>(nested), error_single_dimension_standard(), adaptive_iterations); 
}


template<typename RS, typename CVG>
auto stepper_control_variate(CVG&& cv_generator, RS&& rs) {
    return StepperControlVariate<std::decay_t<RS>,std::decay_t<CVG>>(std::forward<RS>(rs), std::forward<CVG>(cv_generator));
}

template<typename RS, typename CVG>
auto stepper_bins_control_variate(CVG&& cv_generator, RS&& rs) {
    return StepperBinsControlVariate<std::decay_t<RS>,std::decay_t<CVG>>(std::forward<RS>(rs), std::forward<CVG>(cv_generator));
}

template<typename CVG>
auto stepper_control_variate(CVG&& cv_generator, std::size_t seed = std::random_device()()) {
    return stepper_control_variate(std::forward<CVG>(cv_generator),stepper_monte_carlo_uniform(seed));
}

template<typename CVG>
auto stepper_bins_control_variate(CVG&& cv_generator, std::size_t seed = std::random_device()()) {
    return stepper_bins_control_variate(std::forward<CVG>(cv_generator),stepper_bins_per_bin(stepper_monte_carlo_uniform(seed)));
}


