#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
// #include <iostream>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "ecole/exception.hpp"
#include "ecole/scip/model.hpp"

#include "scip/utils.hpp"

namespace ecole {
namespace scip {

template <> void Deleter<SCIP>::operator()(SCIP* scip) {
	scip::call(SCIPfree, &scip);
}

unique_ptr<SCIP> create() {
	SCIP* scip_raw;
	scip::call(SCIPcreate, &scip_raw);
	SCIPmessagehdlrSetQuiet(SCIPgetMessagehdlr(scip_raw), true);
	auto scip_ptr = unique_ptr<SCIP>{};
	scip_ptr.reset(scip_raw);
	return scip_ptr;
}

unique_ptr<SCIP> copy(SCIP const* source) {
	if (!source) return nullptr;
	if (SCIPgetStage(const_cast<SCIP*>(source)) == SCIP_STAGE_INIT) return create();
	auto dest = create();
	// Copy operation is not thread safe
	static std::mutex m{};
	std::lock_guard<std::mutex> g{m};
	scip::call(
		SCIPcopy,
		const_cast<SCIP*>(source),
		dest.get(),
		nullptr,
		nullptr,
		"",
		true,
		false,
		false,
		nullptr);
	return dest;
}

SCIP* Model::get_scip_ptr() const noexcept {
	return scip.get();
}

Model::Model() : scip(create()) {
	scip::call(SCIPincludeDefaultPlugins, get_scip_ptr());
}

Model::Model(unique_ptr<SCIP>&& scip) {
	if (scip)
		this->scip = std::move(scip);
	else
		throw Exception("Cannot create empty model");
}

Model::Model(Model const& other) : scip(copy(other.get_scip_ptr())) {}

Model& Model::operator=(Model const& other) {
	if (&other != this) scip = copy(other.get_scip_ptr());
	return *this;
}

bool Model::operator==(Model const& other) const noexcept {
	return scip == other.scip;
}

bool Model::operator!=(Model const& other) const noexcept {
	return !(*this == other);
}

Stage Model::getStage() {
	switch (SCIPgetStage(get_scip_ptr())) {
		case SCIP_STAGE_INIT:
			return Stage::Init;
		case SCIP_STAGE_PROBLEM:
			return Stage::Problem;
		case SCIP_STAGE_TRANSFORMING:
			return Stage::Transforming;
		case SCIP_STAGE_TRANSFORMED:
			return Stage::Transformed;
		case SCIP_STAGE_INITPRESOLVE:
			return Stage::InitPresolve;
		case SCIP_STAGE_PRESOLVING:
			return Stage::Presolving;
		case SCIP_STAGE_EXITPRESOLVE:
			return Stage::ExitPresolve;
		case SCIP_STAGE_PRESOLVED:
			return Stage::Presolved;
		case SCIP_STAGE_INITSOLVE:
			return Stage::InitSolve;
		case SCIP_STAGE_SOLVING:
			return Stage::Solving;
		case SCIP_STAGE_SOLVED:
			return Stage::Solved;
		case SCIP_STAGE_EXITSOLVE:
			return Stage::ExitSolve;
		case SCIP_STAGE_FREETRANS:
			return Stage::FreeTrans;
		case SCIP_STAGE_FREE:
			return Stage::Free;
		default:
			throw Exception("Unexpected SCIP_STAGE value.");
	}
}

ParamType Model::get_param_type(std::string const & name) const {
	auto* scip_param = SCIPgetParam(get_scip_ptr(), name.c_str());
	if (!scip_param)
		throw Exception(fmt::format("Unknown parameter '{}'", name));
	else
		switch (SCIPparamGetType(scip_param)) {
		case SCIP_PARAMTYPE_BOOL:
			return ParamType::Bool;
		case SCIP_PARAMTYPE_INT:
			return ParamType::Int;
		case SCIP_PARAMTYPE_LONGINT:
			return ParamType::LongInt;
		case SCIP_PARAMTYPE_REAL:
			return ParamType::Real;
		case SCIP_PARAMTYPE_CHAR:
			return ParamType::Char;
		case SCIP_PARAMTYPE_STRING:
			return ParamType::String;
		default:
			assert(false);  // All enum value should be handled
			// Non void return for optimized build
			throw Exception(fmt::format("Unrecognized type for parameter '{}'", name));
		}
}

param_t<ParamType::Int> Model::seed() const {
	return get_param<param_t<ParamType::Int>>("randomization/randomseedshift");
}

template <typename T> static auto mod(T num, T div) noexcept {
	return (num % div + div) % div;
}

void Model::readProb(const std::string& filename) {
	scip::call(SCIPreadProb, get_scip_ptr(), filename.c_str(), nullptr);
}

void Model::seed(param_t<ParamType::Int> seed_v) {
	set_param("randomization/randomseedshift", seed_v);
	set_param("randomization/permutationseed", seed_v);
}

void Model::solve() {
	scip::call(SCIPsolve, get_scip_ptr());
}

void Model::interrupt_solve() {
	scip::call(SCIPinterruptSolve, get_scip_ptr());
}

void Model::disable_presolve() {
	scip::call(SCIPsetPresolving, get_scip_ptr(), SCIP_PARAMSETTING_OFF, true);
}
void Model::disable_cuts() {
	scip::call(SCIPsetSeparating, get_scip_ptr(), SCIP_PARAMSETTING_OFF, true);
}

bool Model::is_solved() const noexcept {
	return SCIPgetStage(get_scip_ptr()) == SCIP_STAGE_SOLVED;
}

VarView Model::variables() const noexcept {
	auto const scip_ptr = get_scip_ptr();
	auto const n_vars = static_cast<std::size_t>(SCIPgetNVars(scip_ptr));
	return VarView(scip_ptr, SCIPgetVars(scip_ptr), n_vars);
}

VarView Model::lp_branch_cands() const noexcept {
	int n_vars{};
	SCIP_VAR** vars{};
	scip::call(
		SCIPgetLPBranchCands,
		get_scip_ptr(),
		&vars,
		nullptr,
		nullptr,
		&n_vars,
		nullptr,
		nullptr);
	return VarView(get_scip_ptr(), vars, static_cast<std::size_t>(n_vars));
}

ColView Model::lp_columns() const {
	auto const scip_ptr = get_scip_ptr();
	if (SCIPgetStage(scip_ptr) != SCIP_STAGE_SOLVING)
		throw Exception("LP columns are only available during solving");
	auto const n_cols = static_cast<std::size_t>(SCIPgetNLPCols(scip_ptr));
	return ColView(scip_ptr, SCIPgetLPCols(scip_ptr), n_cols);
}

RowView Model::lp_rows() const {
	auto const scip_ptr = get_scip_ptr();
	if (SCIPgetStage(scip_ptr) != SCIP_STAGE_SOLVING)
		throw Exception("LP rows are only available during solving");
	auto const n_rows = static_cast<std::size_t>(SCIPgetNLPRows(scip_ptr));
	return RowView(scip_ptr, SCIPgetLPRows(scip_ptr), n_rows);
}

}  // namespace scip
}  // namespace ecole

struct SCIP_BranchruleData {
	ecole::scip::Model::BranchFunc func;
	ecole::scip::Model& model;
};

namespace ecole {
namespace scip {

class Model::LambdaBranchRule {
	// A SCIP branch rule class that runs a given function.
	// The scip BranchRule is actually never substituted, but its internal data is changed
	// to a new function.

private:
	static constexpr auto name = "ecole::scip::LambdaBranchRule";
	static constexpr auto description = "";
	static constexpr auto priority = 536870911;  // Maximum branching rule priority
	static constexpr auto maxdepth = -1;         // No maximum depth
	static constexpr auto maxbounddist = 1.0;    // No distance to dual bound

	static auto exec_lp(
		SCIP* scip,
		SCIP_BRANCHRULE* branch_rule,
		SCIP_Bool allow_addcons,
		SCIP_RESULT* result) {
		// The function that is called to branch on lp fractional varaibles, as required
		// by SCIP.
		(void)allow_addcons;
		auto const branch_data = SCIPbranchruleGetData(branch_rule);
		assert(branch_data->model.get_scip_ptr() == scip);
		assert(branch_data->func);
		*result = SCIP_DIDNOTRUN;

		// C code must be exception safe.
		try {
			auto var = branch_data->func(branch_data->model);
			if (var == VarProxy::None)
				*result = SCIP_DIDNOTRUN;
			else {
				scip::call(SCIPbranchVar, scip, var.value, nullptr, nullptr, nullptr);
				*result = SCIP_BRANCHED;
			}
		} catch (std::exception& e) {
			SCIPerrorMessage(e.what());
			return SCIP_BRANCHERROR;
		} catch (...) {
			return SCIP_BRANCHERROR;
		}
		return SCIP_OKAY;
	}

	static auto include_void_branch_rule(Model& model) {
		auto const scip = model.get_scip_ptr();
		SCIP_BRANCHRULE* branch_rule;
		scip::call(
			SCIPincludeBranchruleBasic,
			scip,
			&branch_rule,
			name,
			description,
			priority,
			maxdepth,
			maxbounddist,
			new SCIP_BranchruleData{Model::BranchFunc{nullptr}, model});
		scip::call(SCIPsetBranchruleExecLp, scip, branch_rule, exec_lp);
		return branch_rule;
	}

	static inline auto get_branch_rule(Model const& model) {
		return SCIPfindBranchrule(model.get_scip_ptr(), name);
	}

	static void
	set_branch_func(SCIP_BRANCHRULE* const branch_rule, Model::BranchFunc const& func) {
		auto const branch_data = SCIPbranchruleGetData(branch_rule);
		branch_data->func = func;
	}

public:
	LambdaBranchRule() = delete;

	static void set_branch_func(Model& model, Model::BranchFunc const& func) {
		auto branch_rule = get_branch_rule(model);
		if (!branch_rule) branch_rule = include_void_branch_rule(model);
		set_branch_func(branch_rule, func);
	}
};

char const* const Model::LambdaBranchRule::name;
char const* const Model::LambdaBranchRule::description;
int const Model::LambdaBranchRule::priority;
int const Model::LambdaBranchRule::maxdepth;
double const Model::LambdaBranchRule::maxbounddist;

void Model::set_branch_rule(BranchFunc const& func) {
	LambdaBranchRule::set_branch_func(*this, func);
}

}  // namespace scip
}  // namespace ecole
