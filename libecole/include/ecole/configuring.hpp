#pragma once

#include <memory>
#include <tuple>
#include <vector>
#include <map>
#include <random>
#include <nonstd/optional.hpp>  // C++17-like optional
#include <nonstd/variant.hpp>  // C++17-like variant
#include <fmt/format.h>  // C++20-like format

#include "ecole/environment/abstract.hpp"
#include "ecole/observation/abstract.hpp"
#include "ecole/reward/abstract.hpp"
#include "ecole/scip/model.hpp"
#include "ecole/termination/abstract.hpp"

using namespace std;
using namespace nonstd;

namespace ecole {
namespace configuring {

using SCIP_Int = int;
using SCIP_Char = char;
using SCIP_String = char const *;

using ParamValue = variant<bool, SCIP_Longint, SCIP_Real, string>;
// using ParamValue = variant<SCIP_Bool, SCIP_Int, SCIP_Longint, SCIP_Real, SCIP_String, SCIP_Char>;

class Observation {
};

using action_t = map<string, ParamValue>;

using reward::Reward;

using obs_done_t = tuple<optional<shared_ptr<Observation>>, bool>;
using obs_rwd_done_t = tuple<optional<shared_ptr<Observation>>, Reward, bool>;

class MyEnv {
public:
	MyEnv(optional<string> const & instance_ = optional<string>(), int seed_ = 0);
	virtual ~MyEnv() = default;
	virtual obs_done_t reset(optional<string> const & instance_file_ = optional<string>(), optional<int> const & seed_ = optional<int>());
	virtual obs_rwd_done_t step(action_t const & action);
	virtual shared_ptr<scip::Model> build_model();

private:

	// 32 bits generator
	using rng_t = mersenne_twister_engine<uint32_t,
		32,624,397,31,0x9908b0df,11,0xffffffff,
		7,0x9d2c5680,15,0xefc60000,18,1812433253>;

	vector<string> const param_names;  // parameters to be acted on
	rng_t seed_rng;  // random number generator for episode seeds

	optional<string> instance;  // instance for the current episode
	int episode_seed;  // seed of the current episode
	rng_t episode_rng;  // internal random number generator of the current episode
	shared_ptr<scip::Model> model;  // model of the current episode
};

template <typename Action> class ActionFunction {
public:
	using action_t = Action;

	virtual ~ActionFunction() = default;
	virtual void set(scip::Model& model, Action const& action) = 0;
	virtual unique_ptr<ActionFunction> clone() const = 0;
};

template <typename Action> class Configure : public ActionFunction<Action> {
public:
	string const param;

	Configure(string param) noexcept;
	void set(scip::Model& model, Action const& action) override;
	virtual unique_ptr<ActionFunction<Action>> clone() const override;
};

template <
	typename Action,
	typename Observation,
	template <typename...> class Holder = unique_ptr>
class Environment : public environment::Environment<Action, Observation, Holder> {
public:
	using env_t = environment::Environment<Action, Observation, Holder>;
	using typename env_t::info_t;
	using typename env_t::seed_t;

	template <typename T> using ptr = typename env_t::template ptr<T>;

	Environment(
		ptr<observation::ObservationFunction<Observation>>&& obs_func,
		ptr<ActionFunction<Action>>&& action_func);

private:
	ptr<scip::Model> _model;
	ptr<observation::ObservationFunction<Observation>> obs_func;
	ptr<ActionFunction<Action>> action_func;

	tuple<Observation, bool> _reset(ptr<scip::Model>&& model) override;
	tuple<Observation, Reward, bool, info_t> _step(Action action) override;
	bool is_done() const noexcept;
};

/***********************************
 *  Implementation of Environment  *
 ***********************************/

template <typename A>
Configure<A>::Configure(string param) noexcept : param(move(param)) {}

template <typename A> void Configure<A>::set(scip::Model& model, A const& action) {
	model.set_param(param, action);
}

template <typename A>
auto Configure<A>::clone() const -> unique_ptr<ActionFunction<A>> {
	return make_unique<Configure<A>>(*this);
}

template <typename A, typename O, template <typename...> class H>
Environment<A, O, H>::Environment(
	ptr<observation::ObservationFunction<O>>&& obs_func,
	ptr<ActionFunction<A>>&& action_func) :
	obs_func(move(obs_func)), action_func(move(action_func)) {}

template <typename A, typename O, template <typename...> class H>
auto Environment<A, O, H>::_reset(ptr<scip::Model>&& model) -> tuple<O, bool> {
	_model = move(model);

	return {obs_func->get(*_model), is_done()};
}

template <typename A, typename O, template <typename...> class H>
auto Environment<A, O, H>::_step(A action) -> tuple<O, Reward, bool, info_t> {
	action_func->set(*_model, action);
	_model->solve();
	return {obs_func->get(*_model), 0., true, info_t{}};
}

template <typename A, typename O, template <typename...> class H>
bool Environment<A, O, H>::is_done() const noexcept {
	return _model->is_solved();
}

}  // namespace configuring
}  // namespace ecole
