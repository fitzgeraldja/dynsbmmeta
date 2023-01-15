\name{compute.icl}
\alias{compute.icl}
\title{
Compute (approximate) Integrated Completed Likelihood (ICL) value for given dynamic SBM, possibly with metadata.
}
\description{
Compute (approximate) Integrated Completed Likelihood (ICL) value for given dynamic SBM, possibly with metadata -- the types of metadata (if any) must be passed as a vector.
}
\usage{
compute.icl(dynsbmmeta, meta.types = c("poisson", "categorical x L", "negative binomial", "(independent/shared) normal", "independent bernoulli x L"))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{dynsbmmeta}{
An object of class \code{dynsbmmeta}, i.e. the model for which we want to calculate ICL.
}
  \item{meta.types}{
Vector listing type for each piece of metadata considered by the model.
}
}
\details{
The value returned is effectively just half the Bayesian Information Criterion (BIC), that is (1/2)klog(n) - L, where k is the number of parameters, n is the number of observations, and L is the log likelihood of the observed data given the parameters.
}
\value{
ICL value.
}
\references{
[PREPRINT UPCOMING]
}
\author{
Authors: John Fitzgerald

Maintainer: John Fitzgerald <fitzgeraldj@maths.ox.ac.uk>
}
\examples{
data(simdataT5Q4N40binary)

## estimation for Q=1..6 groups
list.dynsbmmeta <- select.dynsbmmeta(simdataT5Q4N40binary, 
				Qmin=1, Qmax=6, edge.type="binary", nstart=1)
				
\dontrun{
## better to use nstart>1 starting points
## but estimation can take 1-2 minutes
list.dynsbmmeta <- select.dynsbmmeta(simdataT5Q4N40binary, 
				Qmin=1, Qmax=6, edge.type="binary", nstart=25)
}

## selection of Q=4
dynsbmmeta <- list.dynsbmmeta[[4]]

## calculate ICL
icl <- compute.icl(dynsbmmeta,meta.types=c())
}
