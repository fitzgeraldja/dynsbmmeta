# This file is part of dynsbmmeta.

# dynsbmmeta is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# dynsbmmeta is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with dynsbmmeta.  If not, see <http://www.gnu.org/licenses/>

library(stringr)

select.dynsbmmeta <- function(Y, X=NULL, present=NULL, Qmin, Qmax,
                          edge.type=c("binary","discrete","continuous","poisson"), 
                          meta.present=NULL, 
                          meta.types=c("poisson","categorical x L",
                                       "negative binomial",
                                       "(independent/shared) normal",
                                       "independent bernoulli x L"),
                          meta.dims=NULL, K=-1,
                          directed=FALSE, self.loop=FALSE,
                          meta.tuning=c(1.0),
                          nb.cores=1,
                          iter.max=20, nstart=25, perturbation.rate=0.2,
                          fixed.param=FALSE,
                          bipartition=NULL,
                          plot=TRUE,
                          ret.best=FALSE,
                          return.marginals=FALSE,
                          verbose=TRUE,
                          save.on.run=FALSE,
                          save.path=""){
    if (is.null(present)){
        present <- matrix(0L,dim(Y)[2],dim(Y)[1])    
        for (t in 1:dim(Y)[1])
            present[union(which(apply(Y[t,,],1,FUN=function(v) sum(v>0,na.rm=TRUE))>0),which(apply(Y[t,,],2,FUN=function(v) sum(v>0,na.rm=TRUE))>0)),t] <- 1    		
    }
    never.present <- which(rowSums(present)==0L)
    if(length(never.present)){
        stop("Data format error: one or more nodes are never present.\nThis is not supported (see help about the present argument).\nPlease correct or remove the nodes from the adjacency matrices.")
    }
    if (!is.null(X)){
        if (is.null(meta.present)){
            # create S length list of T x N IntegerMatrix
            # assuming X is length S list of T x N x metadims NumericMatrix
            # if metadata is not available must label as NA explicitly, as real values could be zero
            meta.present<-list()
            for(s in 1:length(X)){
                tempmat<-matrix(0L,dim(Y)[1],dim(Y)[2])
                if (dim(X[[s]])[3]==1){
                    # handle with X[[s]][t,,,drop=F] - means still have t as first dim
                    for (t in 1:dim(Y)[1]){
                        tempmat[t,which(apply(X[[s]][t,,,drop=F],2,FUN=function(v) sum(is.na(v)))==0)] <- 1
                    }
                } else {
                    for (t in 1:dim(Y)[1]){
                        # NB classing metadata as not present if any value is missing, even if some present (to simplify)
                        tempmat[t,which(apply(X[[s]][t,,],1,FUN=function(v) sum(is.na(v)))==0)] <- 1
                    }
                }
                
                meta.present<-append(meta.present,list(tempmat))
            }
            # print(meta.present)
        }
        if (is.null(meta.dims)){
            # L<-strtoi(str_split_fixed(meta.types[s]," x ")[[2]]) 
            # ^unnecessary anyway, could have just take dim(X[[s]])[3]
            meta.dims<-vector(mode="integer", length=length(X))
            if (!length(meta.types)==length(X)){
                stop("Insufficient named distributions for metadata provided")
            }
            for(s in 1:length(X)){
                meta.dims[s] <- dim(X[[s]])[3] # 1st dim of X[[s]] is T, 2nd N, then 3rd metadim
                if (!(meta.types[s] %in% c("poisson",
                                          "negative binomial"
                                          ))
                    &&!grepl("categorical",meta.types[s],fixed=TRUE)&&!grepl("independent bernoulli",meta.types[s],fixed=TRUE)&&!grepl("normal",meta.types[s],fixed=TRUE)){
                    stop("Unknown metadata distribution provided")
                } 
            }
        }
    }
        
    if(!is.null(bipartition)){
        fixed.param <- TRUE
        if(length(bipartition)!=dim(Y)[2]){
            stop("Data format error: bipartition length is uncorrect.")
        }
    }
    list.dynsbmmeta <- list()
    if (!ret.best){
        full.results <- list()
    }
    for (Q in Qmin:Qmax){
        if ((Qmin!=Qmax)&verbose) print(paste("Using Q =",Q))
        results <- list()
        for (rep in 1:nstart){
            if (verbose) print(paste("Fitting model for random initialisation no.",rep))
            if (rep==1) this.perturbation.rate <- 0.0 else this.perturbation.rate <- perturbation.rate
            results[[length(results)+1]] <- estimate.dynsbmmeta(Y=Y,X=X, present=present, Q=Q, 
                                                                meta.present=meta.present,
                                                                meta.types=meta.types,
                                                                meta.dims=meta.dims,
                                                                directed=directed,
                                                            self.loop=self.loop,
                                                            edge.type=edge.type, K=K,
                                                            nb.cores=nb.cores, 
                                                            init.cluster=NULL,
                                                            meta.tuning=meta.tuning,
                                                            perturbation.rate=this.perturbation.rate,
                                                            iter.max=5, fixed.param=fixed.param,
                                                            bipartition=bipartition,
                                                            return.marginals=return.marginals)
            if (save.on.run) save(results,file=save.path)
        }
        if (verbose) print("Finished, now fitting final model using best partition found from random initialisations...")
        best.result <- which.max(sapply(results, FUN=function(result) result$dynsbmmeta$loglikelihood))
        best.init.cluster <- results[[best.result]]$init.cluster
        dynsbmmeta <- estimate.dynsbmmeta(Y=Y, X=X, present=present, Q=Q, 
                                          meta.present=meta.present,
                                          meta.types=meta.types,
                                          meta.dims=meta.dims,
                                          directed=directed,
                                  self.loop=self.loop,
                                  edge.type=edge.type, K=K,
                                  nb.cores=nb.cores,
                                  init.cluster=best.init.cluster,
                                  meta.tuning=meta.tuning,
                                  iter.max=iter.max, fixed.param=fixed.param,
                                  bipartition=bipartition,
                                  return.marginals=return.marginals)$dynsbmmeta
        class(dynsbmmeta) <- c("list","dynsbmmeta")
        list.dynsbmmeta[[length(list.dynsbmmeta)+1]] <- dynsbmmeta
        if ((!ret.best)&(Qmin!=Qmax)){
            results <- lapply(results,function(x) x$dynsbmmeta)
            results[[length(results)+1]] <- dynsbmmeta
            full.results <- append(full.results,results)
        }
        if (verbose) print("Done!")
    }
    if(plot) plot.icl(list.dynsbmmeta,meta.types=meta.types)
    if(ret.best){
        return(list.dynsbmmeta)
    }else{
        if (Qmin==Qmax){
            # return full set of fitted models for box plots - assume Q known
            results <- lapply(results,function(x) x$dynsbmmeta)
            results[[length(results)+1]] <- dynsbmmeta
            return(results) 
        }else{
            return(full.results)
        }
    }
}

plot.icl <- function(list.dynsbmmeta,meta.types=c("poisson","categorical x L",
                                                  "negative binomial",
                                                  "(independent/shared) normal",
                                                  "independent bernoulli x L")){
    Qmin <- ncol(list.dynsbmmeta[[1]]$trans)
    Qmax <- ncol(list.dynsbmmeta[[length(list.dynsbmmeta)]]$trans)
    logl <- sapply(list.dynsbmmeta, FUN=function(model) model$loglikelihood)
    plot(Qmin:Qmax, logl, type='b', ylab="", xlab="Number of groups", yaxt='n')
    par(new=TRUE)
    if("gamma" %in% names(list.dynsbmmeta[[1]])){
        legend("topright",legend=c("Loglikelihood", "ICL\n(not available)"), col=1:2, pch=19, lwd=4)
    } else{
        plot(Qmin:Qmax, sapply(list.dynsbmmeta, FUN=function(model) compute.icl(model, meta.types=meta.types)), col=2, type='b', ylab="", xlab="Number of groups", yaxt='n')
        legend("topright",legend=c("Loglikelihood","ICL"), col=1:2, pch=19, lwd=4)
    }
}

compute.icl <- function(dynsbmmeta,meta.types=c("poisson","categorical x L",
                                                "negative binomial",
                                                "(independent/shared) normal",
                                                "independent bernoulli x L")){    
    T <- ncol(dynsbmmeta$membership)
    Q <- nrow(dynsbmmeta$trans)
    N <- nrow(dynsbmmeta$membership)
    pen <- 0.5*Q*log(N*(N-1)*T/2) + 0.25*Q*(Q-1)*T*log(N*(N-1)/2) # binary case
    # pen <- 0.5*(2*Q)*log(N*(N-1)*T/2) + 0.25*(2*Q*(Q-1)*T)*log(N*(N-1)/2)# poisson case - same as continuous below
    if ("sigma" %in% names(dynsbmmeta)) pen <- 2*pen # continuous case
    if (!is.null(meta.types)){
        sumdistdims <- 0.0
        for (i in 1:length(meta.types)){
            meta.type<-meta.types[[i]]
            if (meta.type=="poisson"){
                sumdistdims<-sumdistdims + 1
            }else if (meta.type=="negative binomial"){
                sumdistdims<-sumdistdims + 2
            }else if ("independent bernoulli"%in%meta.type){
                l<-stoi(str_split_fixed(meta.type," x ",2)[[2]])
                sumdistdims<-sumdistdims + l
            }else if ("categorical"%in%meta.type){
                l<-stoi(str_split_fixed(meta.type," x ",2)[[2]])
                sumdistdims<-sumdistdims + l
            }else if ("normal"%in%meta.type){
                if ("independent"%in%meta.type){
                    Dn <- dim(dynsbmmeta$varphi[[i]])[3]/2
                    sumdistdims <- sumdistdims + 2*Dn 
                } else if ("shared"%in%meta.type){
                    Dn <- dim(dynsbmmeta$varphi[[i]])[3]-1
                    sumdistdims <- sumdistdims + Dn + 1
                } else {
                    Dn <- 0.5*(sqrt(4*dim(dynsbmmeta$varphi[[i]])[3]+1)-1)
                    sumdistdims <- sumdistdims + Dn*(Dn + 1)
                }
            }
        }
        pen <- pen + 0.5*Q*T*(sumdistdims)*log(N) # account for metadata factors
    }
    return(dynsbmmeta$loglikelihood - ifelse(T>1,0.5*Q*(Q-1)*log(N*(T-1)),0) - pen)    
}

infer.metadata <- function(dynsbmmeta,X,i,t,s,meta.type){
    zit=dynsbmmeta$membership[i,t]
    x = X[[s]][t,i,]
    params = dynsbmmeta$varphi[[s]][t,zit,]
    if (meta.type=="poisson"){
        return(list(prob = dpois(x,params), mode_val = floor(params)))
    }else if (meta.type=="negative binomial"){
        return(list(prob = dnbinom(x,params[1],params[2]), mode_val = floor(params[2]*(params[1]-1)/(1-params[2]))))
    }else if ("independent bernoulli"%in%meta.type){
        # NB should really sample and take mode or ask for number of categories to return, as this is just categorical (returning single most likely)
        # could alternatively do MAP and just say 1 if p>0.5 else 0 for each category
        return(list(prob = params[which.max(params)], mode_val = which.max(params)))
    }else if ("categorical"%in%meta.type){
        return(list(prob = params[which.max(params)], mode_val = which.max(params)))
    }else if ("normal"%in%meta.type){
        if ("independent"%in%meta.type){
            ret_probs <- numeric(dim(X[[s]])[3])
            ret_mode <- numeric(dim(X[[s]])[3])
            for (i in 1:length(ret_probs)){
                ret_mode[i] <- params[i]
                ret_probs[i] <- dnorm(params[i],mean=params[i],sd=params[dim(X[[s]])[3]+i]) 
            }
            return(list(prob = prod(ret_probs), mode_val = ret_mode))
        } else if ("shared"%in%meta.type){
            ret_probs <- numeric(dim(X[[s]])[3])
            ret_mode <- numeric(dim(X[[s]])[3])
            for (i in 1:length(ret_probs)){
                ret_mode[i] <- params[i]
                ret_probs[i] <- dnorm(params[i],mean=params[i],sd=params[dim(X[[s]])[3]+1]) 
            }
            return(list(prob = prod(ret_probs), mode_val = ret_mode))
        } else {
            stop("Multivariate normal not implemented.\n")
        }
    }
}

infer.missing.node <- function(dynsbmmeta,use.alpha=FALSE,t,s,meta.data,meta.type){
    # much easier to use group sizes to guide prior p(z_i) as don't pass stationary from model directly (though could find as stationary distribution of transition matrix)
    if (use.alpha){
        stop("Using initial alpha likelihood to infer edges for missing nodes not implemented:\nUsing inferred group sizes")
    }else{
        N<-nrow(dynsbmmeta$membership)
        Q<-ncol(dynsbmmeta$trans)
        z_probs <- numeric(Q)
        pz_x <- numeric(Q)
        sum_prob <- 0.0
        x <- meta.data
        edge.probs <- numeric(N)
        map.edges <- numeric(N)
        for (q in 1:Q){
            n_q=sum(dynsbmmeta$membership[,t]==q)
            z_probs[q]=n_q/N
            params = dynsbmmeta$varphi[[s]][t,q,]
            if (meta.type=="poisson"){
                prob = dpois(x,params)
            }else if (meta.type=="negative binomial"){
                prob = dnbinom(x,params[1],params[2])
            }else if ("independent bernoulli"%in%meta.type){
                # NB should really sample and take mode or ask for number of categories to return, as this is just categorical (returning single most likely)
                # could alternatively do MAP and just say 1 if p>0.5 else 0 for each category
                prob = params[which.max(params)]
            }else if ("categorical"%in%meta.type){
                prob = params[which.max(params)]
            }else if ("normal"%in%meta.type){
                if ("independent"%in%meta.type){
                    ret_probs <- numeric(length(x))
                    for (i in 1:length(ret_probs)){
                        ret_probs[i] <- dnorm(params[i],mean=params[i],sd=params[length(x)+i]) 
                    }
                    prob = prod(ret_probs)
                } else if ("shared"%in%meta.type){
                    ret_probs <- numeric(length(x))
                    for (i in 1:length(ret_probs)){
                        ret_mode[i] <- params[i]
                        ret_probs[i] <- dnorm(params[i],mean=params[i],sd=params[length(x)+1]) 
                    }
                    prob = prod(ret_probs)
                } else {
                    stop("Multivariate normal not implemented.\n")
                }
            }
            pz_x[q] <- prob*z_probs[q]
            sum_prob <- sum_prob + pz_x[q]
        }
        pz_x <- pz_x/sum_prob
        # initially assume undirected binary - fix for general case later
        for (q in 1:Q){
            betas <- dynsbmmeta$beta[t,q,]
            betas <- betas*pz_x[q] # weight by lkl of belonging to group given metadata
            for (j in 1:N){
                zj <- dynsbmmeta$membership[j,t]
                edge.probs[j]<-edge.probs[j]+betas[zj]
            }
        }
        for (j in 1:N){
            if (edge.probs[j]>0.5){
                map.edges[j]<-1.0
            }
        }
        return(map.edges)
    }
}

























