\name{compute.icl}
\alias{compute.icl}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
compute.icl(dynsbmmeta, meta.types = c("poisson", "categorical x L", "negative binomial", "(independent/shared) normal", "independent bernoulli x L"))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{dynsbmmeta}{
%%     ~~Describe \code{dynsbmmeta} here~~
}
  \item{meta.types}{
%%     ~~Describe \code{meta.types} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (dynsbmmeta, meta.types = c("poisson", "categorical x L", 
    "negative binomial", "(independent/shared) normal", "independent bernoulli x L")) 
{
    T <- ncol(dynsbmmeta$membership)
    Q <- nrow(dynsbmmeta$trans)
    N <- nrow(dynsbmmeta$membership)
    pen <- 0.5 * Q * log(N * (N - 1) * T/2) + 0.25 * Q * (Q - 
        1) * T * log(N * (N - 1)/2)
    if ("sigma" \%in\% names(dynsbmmeta)) 
        pen <- 2 * pen
    if (!is.null(meta.types)) {
        sumdistdims <- 0
        for (i in 1:length(meta.types)) {
            meta.type <- meta.types[[i]]
            if (meta.type == "poisson") {
                sumdistdims <- sumdistdims + 1
            }
            else if (meta.type == "negative binomial") {
                sumdistdims <- sumdistdims + 2
            }
            else if ("independent bernoulli" \%in\% meta.type) {
                l <- stoi(str_split_fixed(meta.type, " x ", 2)[[2]])
                sumdistdims <- sumdistdims + l
            }
            else if ("categorical" \%in\% meta.type) {
                l <- stoi(str_split_fixed(meta.type, " x ", 2)[[2]])
                sumdistdims <- sumdistdims + l
            }
            else if ("normal" \%in\% meta.type) {
                if ("independent" \%in\% meta.type) {
                  Dn <- dim(dynsbmmeta$varphi[[i]])[3]/2
                  sumdistdims <- sumdistdims + 2 * Dn
                }
                else if ("shared" \%in\% meta.type) {
                  Dn <- dim(dynsbmmeta$varphi[[i]])[3] - 1
                  sumdistdims <- sumdistdims + Dn + 1
                }
                else {
                  Dn <- 0.5 * (sqrt(4 * dim(dynsbmmeta$varphi[[i]])[3] + 
                    1) - 1)
                  sumdistdims <- sumdistdims + Dn * (Dn + 1)
                }
            }
        }
        pen <- pen + 0.5 * Q * T * (sumdistdims) * log(N)
    }
    return(dynsbmmeta$loglikelihood - ifelse(T > 1, 0.5 * Q * 
        (Q - 1) * log(N * (T - 1)), 0) - pen)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
