// generated using tools::package_native_routine_registration_skeleton(".",character_only = FALSE) after initial compilation
// DO NOT EDIT BY HAND


#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .Call calls */
extern SEXP _dynsbmmeta_dynsbmmetacore(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_dynsbmmeta_dynsbmmetacore", (DL_FUNC) &_dynsbmmeta_dynsbmmetacore, 20},
    {NULL, NULL, 0}
};

void R_init_dynsbmmeta(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
