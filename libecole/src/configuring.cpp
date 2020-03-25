#include <scip/scip.h>

#include "ecole/configuring.hpp"
#include "ecole/exception.hpp"

#include <iostream>

namespace ecole {
namespace configuring {

MyEnv::MyEnv(optional<string> const & instance_, int seed_) { 
	instance = instance_;
	seed_rng.seed(seed_);
	// cout << "Environment seed: " << seed_ << '\n';
}

obs_done_t MyEnv::reset(optional<string> const & instance_, optional<int> const & seed_) {
	// optional: reset instance, otherwise keep the previous one
	if (instance_) {
		instance = instance_;
	}

	// optional: set episode seed, otherwise generate one randomly
	if (seed_) {
		episode_seed = seed_.value();
	}
	else {
		episode_seed = seed_rng();
	}
	// cout << "Episode seed: " << episode_seed << '\n';

	// check for instance file
	if (!instance) {
		throw Exception("The environment requires an instance file, supplied either at construction or in reset().");
	}

	// get a SCIP model (method may be overriden by user)
	model = build_model();

	// assert no instance has been loaded
	if (model->getStage() != scip::Stage::Init) {
		throw Exception("Illegal SCIP model stage, should be in SCIP_STAGE_INIT.");
	}

	// assert model seeds have not been altered
	if (model->get_param<int>("randomization/randomseedshift") != 0) {
		throw Exception("Illegal parameter value for 'randomization/randomseedshift'. This parameter should not be manually set.");
	}
	if (model->get_param<int>("randomization/permutationseed") != 0) {
		throw Exception("Illegal parameter value for 'randomization/randomseedshift'. This parameter should not be manually set.");
	}

	// reset internal random number generator
	episode_rng.seed(episode_seed);

	// reset SCIP internal seed, (31 bits only)
	model->seed(static_cast<int>(episode_rng() >> 1));
	cout << "SCIP seed: " << model->seed() << '\n' << flush;

	// load SCIP instance
	// Note: permutation of the original problem permutation, if any, will
	// happen here. Therefore, permutation parameters and seeds should be
	// set BEFORE loading the instance !
	model->readProb(instance.value());  // 

	return {nullopt, false};
}

obs_rwd_done_t MyEnv::step(action_t const & action) {

	// apply all parameters one by one
    for (auto const & param : action) {
        auto const & name = param.first;
        auto const & value = param.second;
		// cast variant values to their stored type automatically
		visit([this, &name](auto&& typed_value) {
			try {
				cout << fmt::format("Setting parameter '{}' to value '{}' ({}).", name, typed_value, typeid(typed_value).name()) << '\n' << flush;
				model->set_param(name, typed_value);
			} catch (exception const & e) {
				throw_with_nested(Exception(fmt::format("Error while setting parameter '{}' to value '{}'.", name, typed_value)));
			}
        }, value);
    }

	// run model
	model->solve();

	if (!model->is_solved()) {
		throw Exception("Invalid state, should be solved. Was SCIP interrupted ?");
	}

	Reward reward = static_cast<Reward>(-SCIPgetNLPIterations(model->get_scip_ptr()));
	bool done = true;

	if (done) {
		// release model
		model = nullptr;
	}

	return {nullopt, reward, done};
}

shared_ptr<scip::Model> MyEnv::build_model() {
	return make_shared<scip::Model>();
}


}  // namespace configuring
}  // namespace ecole
