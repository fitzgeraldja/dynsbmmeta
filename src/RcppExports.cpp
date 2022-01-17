// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// dynsbmmetacore
List dynsbmmetacore(int T, int N, int Q, int S, NumericVector Yasvector, List Xaslist, const Rcpp::IntegerMatrix& present, std::string edgetype, const Rcpp::List& rmetapresent, const std::vector<std::string>& metatypes, const std::vector<int>& metadims, int K, IntegerVector clustering, const std::vector<double> metatuning, int nbit, int nbthreads, bool isdirected, bool withselfloop, bool frozen, bool ret_marginals);
RcppExport SEXP _dynsbmmeta_dynsbmmetacore(SEXP TSEXP, SEXP NSEXP, SEXP QSEXP, SEXP SSEXP, SEXP YasvectorSEXP, SEXP XaslistSEXP, SEXP presentSEXP, SEXP edgetypeSEXP, SEXP rmetapresentSEXP, SEXP metatypesSEXP, SEXP metadimsSEXP, SEXP KSEXP, SEXP clusteringSEXP, SEXP metatuningSEXP, SEXP nbitSEXP, SEXP nbthreadsSEXP, SEXP isdirectedSEXP, SEXP withselfloopSEXP, SEXP frozenSEXP, SEXP ret_marginalsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type T(TSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< int >::type Q(QSEXP);
    Rcpp::traits::input_parameter< int >::type S(SSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Yasvector(YasvectorSEXP);
    Rcpp::traits::input_parameter< List >::type Xaslist(XaslistSEXP);
    Rcpp::traits::input_parameter< const Rcpp::IntegerMatrix& >::type present(presentSEXP);
    Rcpp::traits::input_parameter< std::string >::type edgetype(edgetypeSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List& >::type rmetapresent(rmetapresentSEXP);
    Rcpp::traits::input_parameter< const std::vector<std::string>& >::type metatypes(metatypesSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type metadims(metadimsSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type clustering(clusteringSEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type metatuning(metatuningSEXP);
    Rcpp::traits::input_parameter< int >::type nbit(nbitSEXP);
    Rcpp::traits::input_parameter< int >::type nbthreads(nbthreadsSEXP);
    Rcpp::traits::input_parameter< bool >::type isdirected(isdirectedSEXP);
    Rcpp::traits::input_parameter< bool >::type withselfloop(withselfloopSEXP);
    Rcpp::traits::input_parameter< bool >::type frozen(frozenSEXP);
    Rcpp::traits::input_parameter< bool >::type ret_marginals(ret_marginalsSEXP);
    rcpp_result_gen = Rcpp::wrap(dynsbmmetacore(T, N, Q, S, Yasvector, Xaslist, present, edgetype, rmetapresent, metatypes, metadims, K, clustering, metatuning, nbit, nbthreads, isdirected, withselfloop, frozen, ret_marginals));
    return rcpp_result_gen;
END_RCPP
}