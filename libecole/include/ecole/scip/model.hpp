#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include <scip/scip.h>

#include "ecole/scip/column.hpp"
#include "ecole/scip/row.hpp"
#include "ecole/scip/variable.hpp"

namespace ecole {
namespace scip {

/* Forward declare scip holder type */
class Scimpl;

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
	Model(Model&&) noexcept;
	Model(Model const& model) = delete;
	Model(std::unique_ptr<Scimpl>&&);

	~Model();

	Model& operator=(Model&&) noexcept;
	Model& operator=(Model const&) = delete;

	/**
	 * Access the underlying SCIP pointer.
	 *
	 * Ownership of the pointer is however not released by the Model.
	 * This function is meant to use the original C API of SCIP.
	 */
	SCIP* get_scip_ptr() const noexcept;

	Model copy_orig() const;

	/**
	 * Compare if two model share the same SCIP pointer, _i.e._ the same memory.
	 */
	bool operator==(Model const& other) const noexcept;
	bool operator!=(Model const& other) const noexcept;

	/**
	 * Construct a model by reading a problem file supported by SCIP (LP, MPS,...).
	 */
	static Model from_file(std::string const& filename);

	/**
	 * Read a problem file into the Model.
	 */
	void read_prob(std::string const& filename);

	Stage get_stage() const noexcept;

	ParamType get_param_type(std::string const& name) const;

	/**
	 * Get and set parameters by their exact SCIP type.
	 *
	 * The method will throw an exception if the type is not *exactly* the one used
	 * by SCIP.
	 *
	 * @see get_param, set_param to convert automatically.
	 */
	template <ParamType T> void set_param_explicit(std::string const& name, param_t<T> value);
	template <ParamType T> param_t<T> get_param_explicit(std::string const& name) const;

	/**
	 * Get and set parameters with automatic casting.
	 *
	 * Often, it is not required to know the exact type of a parameters to set its value
	 * (for instance when setting to zero).
	 * These methods do their best to convert to and from the required type.
	 *
	 * @see get_param_explicit, set_param_explicit to avoid any conversions.
	 */
	template <typename T> void set_param(std::string const& name, T value);
	template <typename T> T get_param(std::string const& name) const;

	void set_params(std::map<std::string, Param> name_values);
	std::map<std::string, Param> get_params() const;

	void disable_presolve();
	void disable_cuts();

	/**
	 * Transform, presolve, and solve problem.
	 */
	void solve();
	bool is_solved() const noexcept;

	void solve_iter();
	void solve_iter_branch(VarProxy var);
	void solve_iter_stop();
	bool solve_iter_is_done();

	VarView variables() const noexcept;
	VarView lp_branch_cands() const noexcept;
	ColView lp_columns() const;
	RowView lp_rows() const;

private:
	std::unique_ptr<Scimpl> scimpl;
};

/*****************************
 *  Implementation of Model  *
 *****************************/

namespace internal {

// SFINAE default class for no available cast
template <typename To, typename From, typename = void> struct Caster {
	static To cast(From) { throw Exception("Cannot convert to the desired type"); }
};

// SFINAE class for available cast
template <typename To, typename From>
struct Caster<To, From, std::enable_if_t<std::is_convertible<From, To>::value>> {
	static To cast(From val) { return static_cast<To>(val); }
};

// Visit From variants.
// Cannot static_cast a variant into one of its held value. Other way around works though.
template <typename To, typename... VariantFrom> struct Caster<To, nonstd::variant<VariantFrom...>> {
	static To cast(nonstd::variant<VariantFrom...> variant_val) {
		return nonstd::visit(
			[](auto val) { return Caster<To, decltype(val)>::cast(val); }, variant_val);
	}
};

// Pointers must not convert to bools
template <typename From> struct Caster<bool, std::remove_cv<From>*> {
	static bool cast(From) { throw Exception("Cannot convert pointers to bool"); }
};

// Convert charachter to string
template <> std::string Caster<std::string, char>::cast(char);

// Convert string to character
template <> char Caster<char, char const*>::cast(char const*);
template <> char Caster<char, std::string>::cast(std::string);

// Helper func to deduce From type automatically
template <typename To, typename From> To cast(From val) {
	return Caster<To, From>::cast(val);
}

}  // namespace internal

template <typename T> void Model::set_param(std::string const& name, T value) {
	using internal::cast;
	switch (get_param_type(name)) {
	case ParamType::Bool:
		return set_param_explicit<ParamType::Bool>(name, cast<bool>(value));
	case ParamType::Int:
		return set_param_explicit<ParamType::Int>(name, cast<int>(value));
	case ParamType::LongInt:
		return set_param_explicit<ParamType::LongInt>(name, cast<long_int>(value));
	case ParamType::Real:
		return set_param_explicit<ParamType::Real>(name, cast<real>(value));
	case ParamType::Char:
		return set_param_explicit<ParamType::Char>(name, cast<char>(value));
	case ParamType::String:
		return set_param_explicit<ParamType::String>(name, cast<std::string>(value));
	default:
		assert(false);  // All enum value should be handled
		// Non void return for optimized build
		throw Exception("Could not find type for given parameter");
	}
}

template <typename T> T Model::get_param(std::string const& name) const {
	using namespace internal;
	switch (get_param_type(name)) {
	case ParamType::Bool:
		return cast<T>(get_param_explicit<ParamType::Bool>(name));
	case ParamType::Int:
		return cast<T>(get_param_explicit<ParamType::Int>(name));
	case ParamType::LongInt:
		return cast<T>(get_param_explicit<ParamType::LongInt>(name));
	case ParamType::Real:
		return cast<T>(get_param_explicit<ParamType::Real>(name));
	case ParamType::Char:
		return cast<T>(get_param_explicit<ParamType::Char>(name));
	case ParamType::String:
		return cast<T>(get_param_explicit<ParamType::String>(name));
	default:
		assert(false);  // All enum value should be handled
		// Non void return for optimized build
		throw Exception("Could not find type for given parameter");
	}
}

}  // namespace scip
}  // namespace ecole
