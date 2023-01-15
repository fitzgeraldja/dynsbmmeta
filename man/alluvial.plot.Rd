\name{alluvial.plot}
\alias{alluvial.plot}
\title{
<<<<<<< HEAD
Alluvial  plot showing  the dynamics  of the  group memberships  associated to a dynamic stochastic block model.
}
\description{
Alluvial  plot showing  the dynamics  of the  group memberships  associated to a dynamic stochastic block model.
}
\usage{
alluvial.plot(dynsbm, timestep.abbrev="T", minimal.flow=1, only.present=FALSE)
}
\arguments{
\item{dynsbm}{An object of class \code{dynsbm} retrieved with the function \code{select.dynsbm}.}
=======
Alluvial  plot showing  the dynamics  of the  group memberships  associated to a dynamic stochastic block model with metadata.
}
\description{
Alluvial  plot showing  the dynamics  of the  group memberships  associated to a dynamic stochastic block model with metadata.
}
\usage{
alluvial.plot(dynsbmmeta, timestep.abbrev="T", minimal.flow=1, only.present=FALSE)
}
\arguments{
\item{dynsbmmeta}{An object of class \code{dynsbmmeta} retrieved with the function \code{select.dynsbmmeta}.}
>>>>>>> dynsbmmeta-cp/master
\item{timestep.abbrev}{Time step abbreviation (D fors days, W for weeks,...).}
\item{minimal.flow}{Minimal flow to be displayed, i.e. minimal number of nodes to consider.}
\item{only.present}{Enable/disable the appearence of the group O which gathers absent nodes.}
}
\details{
Alluvial flows between groups. 
Between any two time plots (on the x-axis), lines are drawn to represent flows between groups.
Each flow corresponds to the membership overlapp between one a group to another group at two consecutive time steps  (represented on the y-axis).  
Each group q at each time step t is called Xt_q where X is time step abbreviation set with \code{timestep.abbrev}. 
The thickness of each line is  proportional  to the  corresponding node counts.
}
\references{
Catherine Matias and Vincent Miele, \emph{Statistical clustering of temporal networks through a dynamic stochastic block model}, Journal of the Royal Statistical Society: Series B (2017)
http://dx.doi.org/10.1111/rssb.12200
http://arxiv.org/abs/1506.07464

Vincent Miele and Catherine Matias, \emph{Revealing the hidden structure
  of dynamic ecological networks},  Royal Society Open Science (2017)
http://dx.doi.org/10.1098/rsos.170251
https://arxiv.org/abs/1701.01355
}
\author{
Authors: Catherine Matias, Vincent Miele
  
<<<<<<< HEAD
Maintainer: Vincent Miele <vincent.miele@univ-lyon1.fr>
=======
Maintainer: John Fitzgerald <fitzgeraldj@maths.ox.ac.uk>
>>>>>>> dynsbmmeta-cp/master
}
\examples{
####################
## 1 - binary case
data(simdataT5Q4N40binary)

## estimation for Q=1..5 groups
<<<<<<< HEAD
list.dynsbm <- select.dynsbm(simdataT5Q4N40binary, 
=======
list.dynsbmmeta <- select.dynsbmmeta(simdataT5Q4N40binary, 
>>>>>>> dynsbmmeta-cp/master
				Qmin=1, Qmax=5, edge.type="binary", nstart=1)
				
\dontrun{
## better to use nstart>1 starting points
## but estimation can take 1-2 minutes
<<<<<<< HEAD
list.dynsbm <- select.dynsbm(simdataT5Q4N40binary, 
=======
list.dynsbmmeta <- select.dynsbmmeta(simdataT5Q4N40binary, 
>>>>>>> dynsbmmeta-cp/master
				Qmin=1, Qmax=5, edge.type="binary", nstart=25)
}

## selection of Q=4
<<<<<<< HEAD
dynsbm <- list.dynsbm[[4]]

## plotting switches between groups
alluvial.plot(dynsbm)
=======
dynsbmmeta <- list.dynsbmmeta[[4]]

## plotting switches between groups
alluvial.plot(dynsbmmeta)
>>>>>>> dynsbmmeta-cp/master

####################
## 2 - continuous case
data(simdataT5Q4N40continuous)

## estimation for Q=1..5 groups
<<<<<<< HEAD
list.dynsbm <- select.dynsbm(simdataT5Q4N40continuous, 
=======
list.dynsbmmeta <- select.dynsbmmeta(simdataT5Q4N40continuous, 
>>>>>>> dynsbmmeta-cp/master
				Qmin=1, Qmax=5, edge.type="continuous", nstart=1)
						
\dontrun{
## better to use nstart>1 starting points
## but estimation can take 1-2 minutes
<<<<<<< HEAD
list.dynsbm <- select.dynsbm(simdataT5Q4N40continuous, 
=======
list.dynsbmmeta <- select.dynsbmmeta(simdataT5Q4N40continuous, 
>>>>>>> dynsbmmeta-cp/master
				Qmin=1, Qmax=5, edge.type="continuous", nstart=25)
}

## selection of Q=4
<<<<<<< HEAD
dynsbm <- list.dynsbm[[4]]

## plotting switches between groups
alluvial.plot(dynsbm)
=======
dynsbmmeta <- list.dynsbmmeta[[4]]

## plotting switches between groups
alluvial.plot(dynsbmmeta)
>>>>>>> dynsbmmeta-cp/master

####################
## 3 - discrete case
data(simdataT5Q4N40discrete)

## estimation for Q=1..5 groups
<<<<<<< HEAD
list.dynsbm <- select.dynsbm(simdataT5Q4N40discrete, 
=======
list.dynsbmmeta <- select.dynsbmmeta(simdataT5Q4N40discrete, 
>>>>>>> dynsbmmeta-cp/master
				Qmin=1, Qmax=5, edge.type="discrete", K=4, nstart=1)
									
\dontrun{
## better to use nstart>1 starting points
## but estimation can take 1-2 minutes
<<<<<<< HEAD
list.dynsbm <- select.dynsbm(simdataT5Q4N40discrete, 
=======
list.dynsbmmeta <- select.dynsbmmeta(simdataT5Q4N40discrete, 
>>>>>>> dynsbmmeta-cp/master
				Qmin=1, Qmax=5, edge.type="discrete", K=4, nstart=25)
}

## selection of Q=4
<<<<<<< HEAD
dynsbm <- list.dynsbm[[4]]

## plotting switches between groups
alluvial.plot(dynsbm)
=======
dynsbmmeta <- list.dynsbmmeta[[4]]

## plotting switches between groups
alluvial.plot(dynsbmmeta)
>>>>>>> dynsbmmeta-cp/master
}