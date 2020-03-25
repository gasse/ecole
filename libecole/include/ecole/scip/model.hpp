#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <iostream>

#include <scip/scip.h>

#include <fmt/format.h>

#include "ecole/scip/column.hpp"
#include "ecole/scip/row.hpp"
#include "ecole/scip/variable.hpp"
#include "scip/utils.hpp"

namespace ecole {
namespace scip {

/**
 * Wrap SCIP pointer free function in a deleter for use with smart pointers.
 */
template <typename T> struct Deleter { void operator()(T* ptr); };
template <typename T> using unique_ptr = std::unique_ptr<T, Deleter<T>>;

/**
 * Create an initialized SCIP pointer without message handler.
 */
unique_ptr<SCIP> create();

enum class Stage {
	Init = SCIP_STAGE_INIT,  // SCIP data structures are initialized, no problem exists
	Problem = SCIP_STAGE_PROBLEM,  // the problem is being created and modified
	Transforming = SCIP_STAGE_TRANSFORMING,   // the problem is being transformed into solving data space
	Transformed = SCIP_STAGE_TRANSFORMED,   // the problem was transformed into solving data space
	InitPresolve = SCIP_STAGE_INITPRESOLVE,   // presolving is initialized
	Presolving = SCIP_STAGE_PRESOLVING,  // the problem is being presolved
	ExitPresolve = SCIP_STAGE_EXITPRESOLVE,  // presolving is exited
	Presolved = SCIP_STAGE_PRESOLVED,  // the problem was presolved
	InitSolve = SCIP_STAGE_INITSOLVE,  // the solving process data is being initialized
	Solving = SCIP_STAGE_SOLVING, // the problem is being solved
	Solved = SCIP_STAGE_SOLVED, // the problem was solved
	ExitSolve = SCIP_STAGE_EXITSOLVE, // the solving process data is being freed
	FreeTrans = SCIP_STAGE_FREETRANS,  // the transformed problem is being freed
	Free = SCIP_STAGE_FREE  // SCIP data structures are being freed
};

/**
 * Types of parameters supported by SCIP.
 *
 * @see param_t to get the associated type.
 */
enum class ParamType {
	Bool = SCIP_PARAMTYPE_BOOL,
	Int = SCIP_PARAMTYPE_INT,
	LongInt = SCIP_PARAMTYPE_LONGINT,
	Real = SCIP_PARAMTYPE_REAL,
	Char = SCIP_PARAMTYPE_CHAR,
	String = SCIP_PARAMTYPE_STRING
};

namespace internal {
// Use with `param_t`.
// File `model.cpp` contains `static_assert`s to ensure this is never out of date
// with SCIP internals.
template <ParamType> struct ParamType_get;
template <> struct ParamType_get<ParamType::Bool> { using type = bool; };
template <> struct ParamType_get<ParamType::Int> { using type = int; };
template <> struct ParamType_get<ParamType::LongInt> { using type = SCIP_Longint; };
template <> struct ParamType_get<ParamType::Real> { using type = SCIP_Real; };
template <> struct ParamType_get<ParamType::Char> { using type = char; };
template <> struct ParamType_get<ParamType::String> { using type = std::string; };
}  // namespace internal

/**
 * Type associated with a ParamType.
 */
template <ParamType T> using param_t = typename internal::ParamType_get<T>::type;

/**
 * A stateful SCIP solver object.
 *
 * A RAII class to manage an underlying `SCIP*`.
 * This is somehow similar to a `pyscipopt.Model`, but with higher level methods
 * tailored for the needs in Ecole.
 * This is the only interface to SCIP in the library.
 */
class Model {
public:
	/**
	 * Construct an *initialized* model with default SCIP plugins.
	 */
	Model();
	Model(unique_ptr<SCIP>&& scip);
	/**
	 * Deep copy the model.
	 */
	Model(Model const& model);
	Model& operator=(Model const&);
	Model(Model&&) noexcept = default;
	Model& operator=(Model&&) noexcept = default;
	~Model() = default;

	/**
	 * Compare if two model share the same SCIP pointer, _i.e._ the same memory.
	 */
	bool operator==(Model const& other) const noexcept;
	bool operator!=(Model const& other) const noexcept;

	ParamType get_param_type(std::string const& name) const;

	/**
	 * Get and set parameters with automatic casting.
	 *
	 * Often, it is not required to know the exact type of a parameters to set its value
	 * (for instance when setting to zero).
	 * These methods do their best to convert to and from the required type.
	 *
	 * @see get_param_explicit, set_param_explicit to avoid any conversions.
	 */

	// specialization for string types
	template <typename T>
	typename std::enable_if<std::is_same<typename std::decay<T>::type, std::string>::value, void>::type
	set_param(std::string const & name, T value);
	// specialization for arithmetic types
	template <typename T>
	typename std::enable_if<std::is_arithmetic<typename std::decay<T>::type>::value, void>::type
	set_param(std::string const & name, T value);
	// specialization for string types
	template <typename T>
	typename std::enable_if<std::is_same<typename std::decay<T>::type, std::string>::value, T>::type
	get_param(std::string const & name) const;
	// specialization for arithmetic types
	template <typename T>
	typename std::enable_if<std::is_arithmetic<typename std::decay<T>::type>::value, T>::type
	get_param(std::string const & name) const;

	/**
	 * Get the current random seed of the Model.
	 */
	param_t<ParamType::Int> seed() const;
	/**
	 * Set the Model random seed shift.
	 *
	 * Set the shift used by with all random seeds in SCIP.
	 * Random seed for individual compenents of SCIP can be set throught the parameters
	 * but will nontheless be shifted by the value set here.
	 * Set a value of zero to disable shiftting.
	 */
	void seed(param_t<ParamType::Int> seed_v);

	void readProb(std::string const& filename);

	void disable_presolve();
	void disable_cuts();

	/**
	 * Transform, presolve, and solve problem.
	 */
	void solve();
	void interrupt_solve();

	Stage getStage();

	bool is_solved() const noexcept;

	VarView variables() const noexcept;
	VarView lp_branch_cands() const noexcept;
	ColView lp_columns() const;
	RowView lp_rows() const;

	using BranchFunc = std::function<VarProxy(Model&)>;
	void set_branch_rule(BranchFunc const& func);

	/**
	 * Access the underlying SCIP pointer.
	 *
	 * Ownership of the pointer is however not released by the Model.
	 * This function is meant to use the original C API of SCIP.
	 */
	SCIP* get_scip_ptr() const noexcept;

private:
	class LambdaBranchRule;
	unique_ptr<SCIP> scip;
};

namespace internal {

template<class Target, class Source>
Target narrow_cast(Source v) {
    auto r = static_cast<Target>(v);
    if (static_cast<Source>(r) != v)
        throw Exception("narrow_cast<>() failed");
    return r;
}

}  // namespace internal

// specialization for string types
template <typename T>
typename std::enable_if<std::is_same<typename std::decay<T>::type, std::string>::value, void>::type
Model::set_param(std::string const & name, T value) {
    using namespace internal;
    auto scip = get_scip_ptr();
	switch (get_param_type(name)) {
	case ParamType::String:
        std::cout << fmt::format("Setting string parameter '{}' to value '{}' ({}).", name, value, typeid(value).name()) << '\n' << std::flush;
        scip::call(SCIPsetStringParam, scip, name.c_str(), value.c_str());
        break;
	case ParamType::Char:
		// accept strings of length 1 as chars
		if (value.length() == 1) {
			set_param(name, value[0]);
			break;
		}
	case ParamType::Bool:
	case ParamType::Int:
	case ParamType::LongInt:
	case ParamType::Real:
		throw Exception(fmt::format("Parameter {} does not accept string values.", name));
		break;
	default:
		assert(false);  // All enum value should be handled
		// Non void return for optimized build
		throw Exception("Could not find type for given parameter");
	}
}

// specialization for arithmetic types
template <typename T>
typename std::enable_if<std::is_arithmetic<typename std::decay<T>::type>::value, void>::type
Model::set_param(std::string const & name, T value) {
    using namespace internal;
    auto scip = get_scip_ptr();
	switch (get_param_type(name)) {
	case ParamType::Bool:
        std::cout << fmt::format("Setting bool parameter '{}' to value '{}' ({}).", name, value, typeid(value).name()) << '\n' << std::flush;
        scip::call(SCIPsetBoolParam, scip, name.c_str(), narrow_cast<param_t<ParamType::Bool>>(value));
        break;
	case ParamType::Int:
        std::cout << fmt::format("Setting int parameter '{}' to value '{}' ({}).", name, value, typeid(value).name()) << '\n' << std::flush;
        scip::call(SCIPsetIntParam, scip, name.c_str(), narrow_cast<param_t<ParamType::Int>>(value));
        break;
	case ParamType::LongInt:
        std::cout << fmt::format("Setting longint parameter '{}' to value '{}' ({}).", name, value, typeid(value).name()) << '\n' << std::flush;
        scip::call(SCIPsetLongintParam, scip, name.c_str(), narrow_cast<param_t<ParamType::LongInt>>(value));
        break;
	case ParamType::Real:
        std::cout << fmt::format("Setting real parameter '{}' to value '{}' ({}).", name, value, typeid(value).name()) << '\n' << std::flush;
        scip::call(SCIPsetRealParam, scip, name.c_str(), narrow_cast<param_t<ParamType::Real>>(value));
        break;
	case ParamType::Char:
        std::cout << fmt::format("Setting char parameter '{}' to value '{}' ({}).", name, value, typeid(value).name()) << '\n' << std::flush;
        scip::call(SCIPsetCharParam, scip, name.c_str(), narrow_cast<param_t<ParamType::Char>>(value));
        break;
	case ParamType::String:
		throw Exception(fmt::format("Parameter {} does not accept numeric values", name));
        break;
	default:
		assert(false);  // All enum value should be handled
		// Non void return for optimized build
		throw Exception("Could not find type for given parameter");
	}
}

// specialization for string types
template <typename T>
typename std::enable_if<std::is_same<typename std::decay<T>::type, std::string>::value, T>::type
Model::get_param(std::string const & name) const {
	using namespace internal;
    auto scip = get_scip_ptr();
	switch (get_param_type(name)) {
	case ParamType::Bool:
	case ParamType::Int:
	case ParamType::LongInt:
	case ParamType::Real:
	case ParamType::Char:
	case ParamType::String:
		throw Exception(fmt::format("Parameter {} does not export into a numeric value", name));
	default:
		assert(false);  // All enum value should be handled
		// Non void return for optimized build
		throw Exception("Could not find type for given parameter");
	}
}

// specialization for arithmetic types
template <typename T>
typename std::enable_if<std::is_arithmetic<typename std::decay<T>::type>::value, T>::type
Model::get_param(std::string const & name) const {
	using namespace internal;
    auto scip = get_scip_ptr();
	switch (get_param_type(name)) {
	case ParamType::Bool:
	{
        SCIP_Bool value;
        scip::call(SCIPgetBoolParam, scip, name.c_str(), &value);
		return narrow_cast<T>(value);
	}
	case ParamType::Int:
	{
        int value;
        scip::call(SCIPgetIntParam, scip, name.c_str(), &value);
		return narrow_cast<T>(value);
	}
	case ParamType::LongInt:
	{
        SCIP_Longint value;
        scip::call(SCIPgetLongintParam, scip, name.c_str(), &value);
		return narrow_cast<T>(value);
	}
	case ParamType::Real:
	{
        SCIP_Real value;
        scip::call(SCIPgetRealParam, scip, name.c_str(), &value);
		return narrow_cast<T>(value);
	}
	case ParamType::Char:
	{
        char value;
        scip::call(SCIPgetCharParam, scip, name.c_str(), &value);
		return narrow_cast<T>(value);
	}
	case ParamType::String:
		throw Exception(fmt::format("Parameter {} does not export into a numeric value", name));
	default:
		assert(false);  // All enum value should be handled
		// Non void return for optimized build
		throw Exception("Could not find type for given parameter");
	}
}

}  // namespace scip
}  // namespace ecole
