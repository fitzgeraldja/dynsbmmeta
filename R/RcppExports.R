# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

dynsbmmetacore <- function(T, N, Q, S, Yasvector, Xaslist, present, edgetype, rmetapresent, metatypes, metadims, K, clustering, metatuning, nbit = 20L, nbthreads = 1L, isdirected = FALSE, withselfloop = FALSE, frozen = FALSE, ret_marginals = FALSE) {
    .Call(`_dynsbmmeta_dynsbmmetacore`, T, N, Q, S, Yasvector, Xaslist, present, edgetype, rmetapresent, metatypes, metadims, K, clustering, metatuning, nbit, nbthreads, isdirected, withselfloop, frozen, ret_marginals)
}

