#include <memory>
#include <string>
#include <nonstd/optional.hpp>  // C++17-like optional
#include <nonstd/variant.hpp>  // C++17-like variant

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <xtensor-python/pytensor.hpp>

// pybind11 automatic conversion for nonstd::optional and nonstd::variant
namespace pybind11 { namespace detail {
    template <typename T>
    struct type_caster<nonstd::optional<T>> : optional_caster<nonstd::optional<T>> {};
    template <typename... Ts>
    struct type_caster<nonstd::variant<Ts...>> : variant_caster<nonstd::variant<Ts...>> {};
}}

#include "ecole/configuring.hpp"
#include "ecole/observation/node-bipartite.hpp"
#include "ecole/scip/model.hpp"

#include "wrapper/environment.hpp"


namespace py = pybind11;

using namespace ecole;
using namespace configuring;


// using ecole::reward::Reward;
// using Action = py::object;

class PyEnvironment : public MyEnv {
public:
    using MyEnv::MyEnv; // Inherit constructors
	obs_done_t reset(optional<string> const & instance_ = optional<string>(), optional<int> const & seed_ = optional<int>())
	override {PYBIND11_OVERLOAD(
		obs_done_t,
		MyEnv,
		reset,
		instance_,
		seed_
	);}
	// obs_rwd_done_t step(vector<ParamValue> const & param_values)
	obs_rwd_done_t step(action_t const & action)
	override {PYBIND11_OVERLOAD(
		obs_rwd_done_t,
		MyEnv,
		step,
		// param_values
		action
	);}
	shared_ptr<scip::Model> build_model()
	override {PYBIND11_OVERLOAD(
		shared_ptr<scip::Model>,
		MyEnv,
		build_model,
	);}
};


PYBIND11_MODULE(configuring, m) {
	m.doc() = "Learning to configure task.";
	// Import of abstract required for resolving inheritance to abstract types
	py11::module abstract_mod = py11::module::import("ecole.abstract");

	py::add_ostream_redirect(m);

	py::class_<MyEnv, PyEnvironment> environment(m, "Environment");
	environment.def(py::init<optional<string> const &, int>(),
		// py::arg("param_names"),
		py::arg("instance") = nullptr,
		py::arg("seed") = 0);
	environment.def("reset", &MyEnv::reset,
		py::arg("instance") = nullptr,
		py::arg("seed") = nullptr);
	environment.def("step", &MyEnv::step,
		py::arg("action").noconvert());  // forbid conversion, e.g., from None values
	// environment.def("step", [](MyEnv & env, py::list list) {
	// 		std::vector<ParamValue> param_values;
	// 		param_values.reserve(list.size());
	// 		for (auto item : list) {
	// 			if (py::isinstance<py::bool_>(item))
	// 				param_values.push_back(ParamValue(item.cast<SCIP_Bool>()));
	// 			else if (py::isinstance<py::int_>(item))
	// 				// Casting to more precise, and may be downcasted in set_param call
	// 				param_values.push_back(ParamValue(item.cast<SCIP_Longint>()));
	// 			else if (py::isinstance<py::float_>(item))
	// 				param_values.push_back(ParamValue(item.cast<SCIP_Real>()));
	// 			else if (py::isinstance<py::str>(item)) {
	// 				// Cast as std::string and let set_param do conversion for char
	// 				param_values.push_back(ParamValue(item.cast<std::string>()));
	// 			} else
	// 				throw Exception("Unexpected parameter type, must be one of {bool, int, float, str}");
	// 		}
	// 		env.step(param_values);
	// 	},
	// 	py::arg("param_values"));
/*

	using ActionFunction = pyenvironment::ActionFunctionBase<configuring::ActionFunction>;
	using Configure = pyenvironment::
		ActionFunction<configuring::Configure<py::object>, configuring::ActionFunction>;
	using Env = pyenvironment::Env<configuring::Environment>;

	py::class_<ActionFunction, std::shared_ptr<ActionFunction>>(m, "ActionFunction");
	py::class_<Configure, ActionFunction, std::shared_ptr<Configure>>(m, "Configure")  //
		.def(py::init<std::string const&>())
		.def("set", [](Configure& c, scip::Model model, py::object param) {
			c.set(model, pyenvironment::Action<py::object>(param));
		});

	py::class_<Env, pyenvironment::EnvBase>(m, "Environment")  //
		.def_static(
			"make_dummy",
			[](std::string const& param) {
				return std::make_unique<Env>(
					std::make_unique<pyobservation::ObsFunction<observation::NodeBipartite>>(),
					std::make_unique<Configure>(param));
			})
		.def(py::init(  //
			[](std::string const& param) {
				return std::make_unique<Env>(
					std::make_unique<pyobservation::ObsFunction<observation::BasicObsFunction>>(),
					std::make_unique<Configure>(param));
			}))
		.def(py::init(  //
			[](
				pyobservation::ObsFunctionBase const& obs_func,
				ActionFunction const& action_func) {
				return std::make_unique<Env>(obs_func.clone(), action_func.clone());
			}))
		.def("step", [](pyenvironment::EnvBase& env, py::object const& action) {
			return env.step(pyenvironment::Action<py::object>(action));
		});
*/
}
