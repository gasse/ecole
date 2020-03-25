#include <scip/scip.h>

#include "conftest.hpp"

ecole::scip::Model get_model() {
	auto model = ecole::scip::Model{};
	model.readProb(problem_file);
	model.disable_cuts();
	model.disable_presolve();
	return model;
}

ScipNoErrorGuard::ScipNoErrorGuard() {
	SCIPmessageSetErrorPrinting(nullptr, nullptr);
}
ScipNoErrorGuard::~ScipNoErrorGuard() {
	SCIPmessageSetErrorPrintingDefault();
}
