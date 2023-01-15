/*
<<<<<<< HEAD
  This file is part of dynsbm.

  dysbm is free software: you can redistribute it and/or modify
=======
  This file is part of dynsbmmeta.

  dynsbmmeta is free software: you can redistribute it and/or modify
>>>>>>> dynsbmmeta-cp/master
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

<<<<<<< HEAD
  dynsbm is distributed in the hope that it will be useful,
=======
  dynsbmmeta is distributed in the hope that it will be useful,
>>>>>>> dynsbmmeta-cp/master
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
<<<<<<< HEAD
  along with dynsbm.  If not, see <http://www.gnu.org/licenses/>
*/
#include<Rcpp.h>
#include<DynSBMBinary.h>
#include<DynSBMDiscrete.h>
#include<DynSBMGaussian.h>
#include<EM.h>
#include<string>
#include<iostream>
=======
  along with dynsbmmeta.  If not, see <http://www.gnu.org/licenses/>
*/




#include<Rcpp.h>

// [[Rcpp::depends(BH)]]

#include<DynSBMBinary.h>
#include<DynSBMDiscrete.h>
#include<DynSBMGaussian.h>
#include<DynSBMPoisson.h>
#include<DynDCSBMPoisson.h>
#include<EM.h>
#include<string>
#include<algorithm>
#include<iostream>
#include <stdexcept>
>>>>>>> dynsbmmeta-cp/master

using namespace dynsbm;
using namespace Rcpp;
using namespace std;
<<<<<<< HEAD
=======

>>>>>>> dynsbmmeta-cp/master
#ifdef _OPENMP
#include<omp.h>
#endif

<<<<<<< HEAD
// [[Rcpp::export]]
List dynsbmcore(int T, int N, int Q,
		NumericVector Yasvector, const Rcpp::IntegerMatrix & present,
		std::string edgetype, int K,
		IntegerVector clustering,
=======
// [[Rcpp::plugins(openmp)]]

// [[Rcpp::export]]
List dynsbmmetacore(int T, int N, int Q, int S,
		NumericVector Yasvector, 
		List Xaslist, // pass metadata as list of S NumericVectors - assume cast DsxTxN (Ds dim of metadata s) as vector
		const Rcpp::IntegerMatrix & present,
		std::string edgetype, 
    const Rcpp::List & rmetapresent, // pass metapresent as list of S T x N matrices
		const std::vector<std::string> & metatypes,
    const std::vector<int> & metadims,
    int K,
		IntegerVector clustering,
		const std::vector<double> metatuning = std::vector<double>(1,1.0), // vector of metadata importance params in (0,1) - if len 1 then global, else must be len Q
>>>>>>> dynsbmmeta-cp/master
		int nbit = 20,
		int nbthreads = 1,
		bool isdirected = false,
		bool withselfloop = false,
<<<<<<< HEAD
		bool frozen = false) {
=======
		bool frozen = false,
    bool ret_marginals = false) {
>>>>>>> dynsbmmeta-cp/master
#ifdef _OPENMP
  omp_set_num_threads(nbthreads);
#endif
  ////////////////////////
  ////////////////////////
<<<<<<< HEAD
  if (edgetype=="binary"){
    EM<DynSBMBinary,int> em(T,N,Q,present,isdirected,withselfloop);
=======
  // for(auto & metatype : metatypes){    
  //   Rcpp::Rcout<< metatype << "\n";
  // }
  // Rcpp::Rcout << metatypes.size() << "\n";
  // if ((metatypes.size()!=S)&&(metatypes.size()!=1)){
  //   throw std::invalid_argument("Insufficient number of distribution names for metadata provided:\nProvide one name if generic, or the same number as types of metadata");
  // }
  // if (metadims.size()!=S){
  //   throw std::invalid_argument("Insufficient specifications of metadata dimensions");
  // }
  // if (Xaslist.size()!=S){
  //   throw std::invalid_argument("Number of pieces of metadata provided does not match S provided:\nCheck input");
  // }
  // NB by allowing mixed types need to handle differently to edgetype, where can separate into unique functions

  if (!metatuning.empty()){
    if (*std::min_element(metatuning.begin(),metatuning.end())<0.0){
      Rcpp::Rcout << "Warning, provided negative tuning parameter...\n";
    }
    if (*std::max_element(metatuning.begin(),metatuning.end())>1.0){
      Rcpp::Rcout << "Warning, provided tuning parameter greater than 1.0...\n";
    }
    if (metatuning.size()>1){
      Rcpp::Rcout << "Group-level tuning parameters not yet implemented.\n";
    }
  }

  std::vector<std::vector<std::vector<int>>> metapresent(S,std::vector<std::vector<int>>(T,std::vector<int>(N,0)));
  for (int spos=0;spos<S;spos++){
    Rcpp::IntegerMatrix metamat = rmetapresent[spos];
    // Rcpp::Rcout << "metamat " << spos << " has shape (" << metamat.nrow() << "," << metamat.ncol() << ")";
    for (int t=0;t<T;t++){
      for (int i=0;i<N;i++){
        metapresent[spos][t][i]=metamat(t,i);
      }
    }
  }

  if (edgetype=="binary"){

    EM<DynSBMBinary,int, std::vector<double> > em(T,N,Q,S,present,metapresent,metatypes,metadims,metatuning,isdirected,withselfloop);
>>>>>>> dynsbmmeta-cp/master
    int*** Y;
    allocate3D<int>(Y,T,N,N);
    int p=0;
    for(int j=0; j<N; j++){
      for(int i=0; i<N; i++){
<<<<<<< HEAD
	for(int t=0; t<T; t++){
	  Y[t][i][j] = int(Yasvector[p]);
	  p++;
	}
      }
    }
    em.initialize(as<vector<int> >(clustering),Y,frozen);
    int nbiteff = em.run(Y,nbit,10,frozen);
=======
        for(int t=0; t<T; t++){
          Y[t][i][j] = int(Yasvector[p]);
          p++;
        }
      }
    }
    // NB need similar for Xaslist, and will need for each subsequent case
    std::vector<double>*** X;
    allocate3Dvectors<double>(X,T,N,S,metadims);    
    for (int spos=0; spos<S; spos++){
      Rcpp::NumericVector Xsvector = Xaslist[spos];
      int p=0;
      for (int ds=0;ds<metadims[spos];ds++){
        for (int i=0;i<N;i++){
          for (int t=0;t<T;t++){
            X[t][i][spos][ds]=Xsvector[p];
            p++;
            if (metapresent[spos][t][i]!=0){
              for (const auto& x : X[t][i][spos]){
                if (std::isnan(x)){
                  throw std::invalid_argument("Loading nan where metadata should be present: (t,i,s) = (" + std::to_string(t) +","+std::to_string(i) +","+std::to_string(spos)+") and p is "+std::to_string(p));
                }
              }
            }
          }
        }
      }
    }
    // Rcpp::Rcout << "Successfully loaded metadata, here there are no nans when metapresent..." << "\n";




    
    em.initialize(as<vector<int> >(clustering),Y,X,frozen);
    int nbiteff = em.run(Y,X,nbit,10,frozen);
>>>>>>> dynsbmmeta-cp/master
    NumericMatrix trans(Q,Q);
    for(int q=0;q<Q;q++) for(int l=0;l<Q;l++) trans[l+q*Q] = em.getModel().getTrans(q,l);
    IntegerMatrix membership(N,T);
    for(int t=0;t<T;t++){
      std::vector<int> groups = em.getModel().getGroupsByMAP(t);
      for(int i=0;i<N;i++) membership[i+t*N] = groups[i]+1;
    }
<<<<<<< HEAD
=======

    Rcpp::NumericVector fintaumdims(3);
    fintaumdims[0] = T; 
    fintaumdims[1] = N; 
    fintaumdims[2] = Q;
    Rcpp::Dimension fintaud(fintaumdims); // get the dim object
    Rcpp::NumericVector fintaums(fintaud); // create vec. with correct dims
    if (ret_marginals){
      // return estimates of marginals
      for(int t=0;t<T;t++){
        for(int i=0; i<N; i++){
          for(int q=0;q<Q;q++){
            fintaums[q*(N*T)+i*T+t]=em.getModel().getfinTaum(t,i,q);
          }
        }
      }
    }

>>>>>>> dynsbmmeta-cp/master
    Rcpp::NumericVector betadims(3);
    betadims[0] = T; betadims[1] = Q; betadims[2] = Q;
    Rcpp::Dimension d(betadims); // get the dim object
    Rcpp::NumericVector beta(d);  // create vec. with correct dims
    for(int t=0;t<T;t++){
      for(int q=0;q<Q;q++){
<<<<<<< HEAD
	for(int l=0;l<Q;l++){
	  beta[l*(Q*T)+q*T+t]= 1-(em.getModel().getBeta(t,q,l)); // cf. paper
	}}}
    double lkl = em.getModel().modelselectionLoglikelihood(Y);
    deallocate3D<int>(Y,T,N,Q);
    return List::create(Rcpp::Named("trans") = trans,
			Rcpp::Named("membership") = membership,
			Rcpp::Named("beta") = beta,
			Rcpp::Named("loglikelihood") = lkl,
			Rcpp::Named("iter") = nbiteff,
			Rcpp::Named("directed") = isdirected,
			Rcpp::Named("self.loop") = withselfloop);
=======
        for(int l=0;l<Q;l++){
          beta[l*(Q*T)+q*T+t]= 1-(em.getModel().getBeta(t,q,l)); // cf. paper
	  }}}

    // Rcpp::Rcout << "Preparing varphi for R" << "\n";

    std::vector<int> distdims = em.getModel().getDistdims();
    Rcpp::List varphi(S); // create list to return meta params
    for (int s=0;s<S;s++){
      Rcpp::NumericVector varphisdims(3);
      varphisdims[0]=T; 
      varphisdims[1]=Q; 
      varphisdims[2]=distdims[s];
      Rcpp::Dimension d(varphisdims); // get the dim object
      Rcpp::NumericVector varphis(d); // create vec. with correct dims
      for(int t=0;t<T;t++){
        for(int q=0;q<Q;q++){
          std::vector<double> temp = em.getModel().getVarphi(t,q,s);
          
          // Rcpp::Rcout << "Varphi for " << t << ", " << q << ", " << s << ":\n";
          // for (auto & val : temp){
          //   Rcpp::Rcout << val << "\n";
          // }
          
          for(int ds=0;ds<distdims[s];ds++){
            varphis[ds*(Q*T)+q*T+t]=temp[ds];
          }
        }
      }
      // Rcpp::Rcout << "varphi[" << s << "]:\n";
      // for(auto& val : varphis){
      //   Rcpp::Rcout << val << "\n";
      // }
      varphi[s]=varphis;
    }
    // Rcpp::Rcout << "Size of varphi list: " << varphi.size() << "\n";
    // Rcpp::Rcout << "Successfully passed varphi to suitable R format" << "\n";
    
    double lkl = em.getModel().modelselectionLoglikelihood(Y, X);

    deallocate3D<int>(Y,T,N,Q);
    deallocate3D<std::vector<double>>(X,T,N,S);
    if (!ret_marginals){
      return List::create(Rcpp::Named("trans") = trans,
        Rcpp::Named("membership") = membership,
        Rcpp::Named("beta") = beta,
        Rcpp::Named("loglikelihood") = lkl,
        Rcpp::Named("iter") = nbiteff,
        Rcpp::Named("directed") = isdirected,
        Rcpp::Named("self.loop") = withselfloop,
        Rcpp::Named("varphi") = varphi);
    }else{
      return List::create(Rcpp::Named("trans") = trans,
        Rcpp::Named("membership") = membership,
        Rcpp::Named("beta") = beta,
        Rcpp::Named("loglikelihood") = lkl,
        Rcpp::Named("iter") = nbiteff,
        Rcpp::Named("directed") = isdirected,
        Rcpp::Named("self.loop") = withselfloop,
        Rcpp::Named("varphi") = varphi,
        Rcpp::Named("fin.taum") = fintaums);
    }
>>>>>>> dynsbmmeta-cp/master
  } else{
    ////////////////////////
    ////////////////////////
    if (edgetype=="discrete"){
<<<<<<< HEAD
      EM<DynSBMDiscrete,int> em(T,N,Q,present,isdirected,withselfloop);
=======
      EM<DynSBMDiscrete,int,std::vector<double>> em(T,N,Q,S,present,metapresent,metatypes,metadims,metatuning,isdirected,withselfloop);
>>>>>>> dynsbmmeta-cp/master
      const_cast<DynSBMDiscrete&>(em.getModel()).setK(K);
      int*** Y;
      allocate3D<int>(Y,T,N,N);
      int p=0;
      for(int j=0; j<N; j++){
<<<<<<< HEAD
	for(int i=0; i<N; i++){
	  for(int t=0; t<T; t++){
	    Y[t][i][j] = int(Yasvector[p]);
	    p++;
	  }
	}
      }
      em.initialize(as<vector<int> >(clustering),Y,frozen);
      int nbiteff = em.run(Y,nbit,10,frozen);
=======
        for(int i=0; i<N; i++){
          for(int t=0; t<T; t++){
            Y[t][i][j] = int(Yasvector[p]);
            p++;
          }
        }
      }
      
      std::vector<double>*** X;
      allocate3Dvectors<double>(X,T,N,S,metadims);       
      for (int spos=0; spos<S; spos++){
        Rcpp::NumericVector Xsvector = Xaslist[spos];
        int p=0;
        for (int ds=0;ds<metadims[spos];ds++){
          for (int i=0;i<N;i++){
            for (int t=0;t<T;t++){
              X[t][i][spos][ds]=Xsvector[p];
              p++;
            }
          }
        }
      }
      
      em.initialize(as<vector<int> >(clustering),Y,X,frozen);
      int nbiteff = em.run(Y,X,nbit,10,frozen);
>>>>>>> dynsbmmeta-cp/master
      NumericMatrix trans(Q,Q);
      for(int q=0;q<Q;q++) for(int l=0;l<Q;l++) trans[l+q*Q] = em.getModel().getTrans(q,l);
      IntegerMatrix membership(N,T);
      for(int t=0;t<T;t++){
<<<<<<< HEAD
	std::vector<int> groups = em.getModel().getGroupsByMAP(t);
	for(int i=0;i<N;i++) membership[i+t*N] = groups[i]+1;
=======
        std::vector<int> groups = em.getModel().getGroupsByMAP(t);
        for(int i=0;i<N;i++) membership[i+t*N] = groups[i]+1;
>>>>>>> dynsbmmeta-cp/master
      }
      Rcpp::NumericVector betadims(3);
      betadims[0] = T; betadims[1] = Q; betadims[2] = Q;
      Rcpp::Dimension d(betadims); // get the dim object
      Rcpp::NumericVector beta(d);  // create vec. with correct dims
      for(int t=0;t<T;t++){
        for(int q=0;q<Q;q++){
          for(int l=0;l<Q;l++){
            beta[l*(Q*T)+q*T+t]= 1-em.getModel().getBeta(t,q,l); // cf. paper
<<<<<<< HEAD
	  }}}
=======
	    }}}
>>>>>>> dynsbmmeta-cp/master
      Rcpp::NumericVector gammadims(4);
      gammadims[0] = T; gammadims[1] = Q; gammadims[2] = Q; gammadims[3] = K;
      Rcpp::Dimension d2(gammadims);                // get the dim object
      Rcpp::NumericVector gamma(d2);             // create vec. with correct dims
      for(int t=0;t<T;t++){
        for(int q=0;q<Q;q++){
          for(int l=0;l<Q;l++){
            for(int k=0;k<K;k++){
              gamma[k*Q*Q*T+l*(Q*T)+q*T+t]= em.getModel().getMultinomproba(t,q,l,k);
	    }}}}
<<<<<<< HEAD
      double lkl = em.getModel().modelselectionLoglikelihood(Y);
      deallocate3D<int>(Y,T,N,Q);
      return List::create(Rcpp::Named("trans") = trans,
=======
      
      std::vector<int> distdims = em.getModel().getDistdims();
      Rcpp::List varphi(S); 
      for (int s=0;s<S;s++){
        Rcpp::NumericVector varphisdims(3);
        varphisdims[0]=T; varphisdims[1]=Q; varphisdims[2]=distdims[s];
        Rcpp::Dimension d(varphisdims);
        Rcpp::NumericVector varphis(d);
        for(int t=0;t<T;t++){
          for(int q=0;q<Q;q++){
            const std::vector<double> & temp = em.getModel().getVarphi(t,q,s);
            for(int ds=0;ds<distdims[s];ds++){
              varphis[ds*(Q*T)+q*T+t]=temp[ds];
            }
          }
        }
        varphi[s]=varphis;
      }

      Rcpp::NumericVector fintaumdims(3);
      fintaumdims[0] = T; 
      fintaumdims[1] = N; 
      fintaumdims[2] = Q;
      Rcpp::Dimension fintaud(fintaumdims); // get the dim object
      Rcpp::NumericVector fintaums(fintaud); // create vec. with correct dims
      if (ret_marginals){
        // return estimates of marginals
        for(int t=0;t<T;t++){
          for(int i=0; i<N; i++){
            for(int q=0;q<Q;q++){
              fintaums[q*(N*T)+i*T+t]=em.getModel().getfinTaum(t,i,q);
            }
          }
        }
      }

      double lkl = em.getModel().modelselectionLoglikelihood(Y, X);

      
      deallocate3D<int>(Y,T,N,Q);
      deallocate3D<std::vector<double>>(X,T,N,S);
      if (!ret_marginals){
        return List::create(Rcpp::Named("trans") = trans,
			  Rcpp::Named("membership") = membership,
			  Rcpp::Named("beta") = beta,
			  Rcpp::Named("gamma") = gamma,
			  Rcpp::Named("loglikelihood") = lkl,
			  Rcpp::Named("iter") = nbiteff,
			  Rcpp::Named("directed") = isdirected,
			  Rcpp::Named("self.loop") = withselfloop,
        Rcpp::Named("varphi") = varphi);
      } else {
        return List::create(Rcpp::Named("trans") = trans,
>>>>>>> dynsbmmeta-cp/master
			  Rcpp::Named("membership") = membership,
			  Rcpp::Named("beta") = beta,
			  Rcpp::Named("gamma") = gamma,
			  Rcpp::Named("loglikelihood") = lkl,
			  Rcpp::Named("iter") = nbiteff,
			  Rcpp::Named("directed") = isdirected,
<<<<<<< HEAD
			  Rcpp::Named("self.loop") = withselfloop);
    } else{
      ////////////////////////
      ////////////////////////
      // (edgetype=="continuous"){
      EM<DynSBMGaussian,double> em(T,N,Q,present,isdirected,withselfloop);
=======
			  Rcpp::Named("self.loop") = withselfloop,
        Rcpp::Named("varphi") = varphi,
        Rcpp::Named("fin.taum") = fintaums);
      }
      
    } else if (edgetype=="continuous"){
      EM<DynSBMGaussian,double,std::vector<double>> em(T,N,Q,S,present,metapresent,metatypes,metadims,metatuning,isdirected,withselfloop);
>>>>>>> dynsbmmeta-cp/master
      double*** Y;
      allocate3D<double>(Y,T,N,N);
      int p=0;
      for(int j=0; j<N; j++){
<<<<<<< HEAD
	for(int i=0; i<N; i++){
	  for(int t=0; t<T; t++){
	    Y[t][i][j] = double(Yasvector[p]);
	    p++;
	  }
	}
      }
      em.initialize(as<vector<int> >(clustering),Y,frozen);
      int nbiteff = em.run(Y,nbit,10,frozen);
=======
        for(int i=0; i<N; i++){
          for(int t=0; t<T; t++){
            Y[t][i][j] = double(Yasvector[p]);
            p++;
          }
        }
      }
      
      std::vector<double>*** X;
      allocate3Dvectors<double>(X,T,N,S,metadims);    
      for (int spos=0; spos<S; spos++){
        Rcpp::NumericVector Xsvector = Xaslist[spos];
        int p=0;
        for (int ds=0;ds<metadims[spos];ds++){
          for (int i=0;i<N;i++){
            for (int t=0;t<T;t++){
              X[t][i][spos][ds]=Xsvector[p];
              p++;
            }
          }
        }
      }
      
      em.initialize(as<vector<int> >(clustering),Y,X,frozen);
      int nbiteff = em.run(Y,X,nbit,10,frozen);
>>>>>>> dynsbmmeta-cp/master
      NumericMatrix trans(Q,Q);
      for(int q=0;q<Q;q++) for(int l=0;l<Q;l++) trans[l+q*Q] = em.getModel().getTrans(q,l);
      IntegerMatrix membership(N,T);
      for(int t=0;t<T;t++){
<<<<<<< HEAD
	std::vector<int> groups = em.getModel().getGroupsByMAP(t);
	for(int i=0;i<N;i++) membership[i+t*N] = groups[i]+1;
=======
        std::vector<int> groups = em.getModel().getGroupsByMAP(t);
        for(int i=0;i<N;i++) membership[i+t*N] = groups[i]+1;
>>>>>>> dynsbmmeta-cp/master
      }
      Rcpp::NumericVector betadims(3);
      betadims[0] = T; betadims[1] = Q; betadims[2] = Q;
      Rcpp::Dimension d(betadims); // get the dim object
      Rcpp::NumericVector beta(d);  // create vec. with correct dims
      for(int t=0;t<T;t++){
        for(int q=0;q<Q;q++){
          for(int l=0;l<Q;l++){
            beta[l*(Q*T)+q*T+t]= 1-(em.getModel().getBeta(t,q,l)); // cf. paper
<<<<<<< HEAD
	  }}}
=======
	    }}}
>>>>>>> dynsbmmeta-cp/master
      Rcpp::NumericVector mu(d);  // create vec. with correct dims
      for(int t=0;t<T;t++){
        for(int q=0;q<Q;q++){
          for(int l=0;l<Q;l++){
            mu[l*(Q*T)+q*T+t]= em.getModel().getMu(t,q,l);
<<<<<<< HEAD
	  }}}
      NumericVector sigma(T);
      for(int t=0;t<T;t++) sigma[t] = em.getModel().getSigma(t);
      double lkl = em.getModel().modelselectionLoglikelihood(Y);
      deallocate3D<double>(Y,T,N,Q);
      return List::create(Rcpp::Named("trans") = trans,
=======
	    }}}
      NumericVector sigma(T);
      for(int t=0;t<T;t++) sigma[t] = em.getModel().getSigma(t);

      std::vector<int> distdims = em.getModel().getDistdims();
      Rcpp::List varphi(S); 
      for (int s=0;s<S;s++){
        Rcpp::NumericVector varphisdims(3);
        varphisdims[0]=T; varphisdims[1]=Q; varphisdims[2]=distdims[s];
        Rcpp::Dimension d(varphisdims);
        Rcpp::NumericVector varphis(d);
        for(int t=0;t<T;t++){
          for(int q=0;q<Q;q++){
            const std::vector<double> & temp = em.getModel().getVarphi(t,q,s);
            for(int ds=0;ds<distdims[s];ds++){
              varphis[ds*(Q*T)+q*T+t]=temp[ds];
            }
          }
        }
        varphi[s]=varphis;
      }

      Rcpp::NumericVector fintaumdims(3);
      fintaumdims[0] = T; 
      fintaumdims[1] = N; 
      fintaumdims[2] = Q;
      Rcpp::Dimension fintaud(fintaumdims); // get the dim object
      Rcpp::NumericVector fintaums(fintaud); // create vec. with correct dims
      if (ret_marginals){
        // return estimates of marginals
        for(int t=0;t<T;t++){
          for(int i=0; i<N; i++){
            for(int q=0;q<Q;q++){
              fintaums[q*(N*T)+i*T+t]=em.getModel().getfinTaum(t,i,q);
            }
          }
        }
      }      

      double lkl = em.getModel().modelselectionLoglikelihood(Y, X);
      
      deallocate3D<double>(Y,T,N,Q);
      deallocate3D<std::vector<double>>(X,T,N,S);
      
      if (!ret_marginals){
        return List::create(Rcpp::Named("trans") = trans,
>>>>>>> dynsbmmeta-cp/master
			  Rcpp::Named("membership") = membership,
			  Rcpp::Named("beta") = beta,
			  Rcpp::Named("mu") = mu,
			  Rcpp::Named("sigma") = sigma,
			  Rcpp::Named("loglikelihood") = lkl,
			  Rcpp::Named("iter") = nbiteff,
			  Rcpp::Named("directed") = isdirected,
<<<<<<< HEAD
			  Rcpp::Named("self.loop") = withselfloop);
=======
			  Rcpp::Named("self.loop") = withselfloop,
        Rcpp::Named("varphi") = varphi);
      } else {
        return List::create(Rcpp::Named("trans") = trans,
			  Rcpp::Named("membership") = membership,
			  Rcpp::Named("beta") = beta,
			  Rcpp::Named("mu") = mu,
			  Rcpp::Named("sigma") = sigma,
			  Rcpp::Named("loglikelihood") = lkl,
			  Rcpp::Named("iter") = nbiteff,
			  Rcpp::Named("directed") = isdirected,
			  Rcpp::Named("self.loop") = withselfloop,
        Rcpp::Named("varphi") = varphi,
        Rcpp::Named("fin.taum") = fintaums);
      }
      
    } else if (edgetype=="poisson"){
      EM<DynSBMPoisson,int,std::vector<double>> em(T,N,Q,S,present,metapresent,metatypes,metadims,metatuning,isdirected,withselfloop);
      int*** Y;
      allocate3D<int>(Y,T,N,N);
      int p=0;
      for(int j=0; j<N; j++){
        for(int i=0; i<N; i++){
          for(int t=0; t<T; t++){
            Y[t][i][j] = int(Yasvector[p]);
            p++;
          }
        }
      }
      
      std::vector<double>*** X;
      allocate3Dvectors<double>(X,T,N,S,metadims);    
      for (int spos=0; spos<S; spos++){
        Rcpp::NumericVector Xsvector = Xaslist[spos];
        int p=0;
        for (int ds=0;ds<metadims[spos];ds++){
          for (int i=0;i<N;i++){
            for (int t=0;t<T;t++){
              X[t][i][spos][ds]=Xsvector[p];
              p++;
              if (metapresent[spos][t][i]!=0){
                for (const auto& x : X[t][i][spos]){
                  if (std::isnan(x)){
                    throw std::invalid_argument("Loading nan where metadata should be present: (t,i,s) = (" + std::to_string(t) +","+std::to_string(i) +","+std::to_string(spos)+") and p is "+std::to_string(p));
                  }
                }
              }
            }
          }
        }
      }
      // Rcpp::Rcout << "Successfully loaded metadata, here there are no nans when metapresent..." << "\n";
      
      em.initialize(as<vector<int> >(clustering),Y,X,frozen);
      int nbiteff = em.run(Y,X,nbit,10,frozen);
      NumericMatrix trans(Q,Q);
      for(int q=0;q<Q;q++) for(int l=0;l<Q;l++) trans[l+q*Q] = em.getModel().getTrans(q,l);
      IntegerMatrix membership(N,T);
      for(int t=0;t<T;t++){
        std::vector<int> groups = em.getModel().getGroupsByMAP(t);
        for(int i=0;i<N;i++) membership[i+t*N] = groups[i]+1;
      }
      Rcpp::NumericVector betadims(3);
      betadims[0] = T; betadims[1] = Q; betadims[2] = Q;
      Rcpp::Dimension d(betadims); // get the dim object
      Rcpp::NumericVector beta(d);  // create vec. with correct dims
      for(int t=0;t<T;t++){
        for(int q=0;q<Q;q++){
          for(int l=0;l<Q;l++){
            beta[l*(Q*T)+q*T+t]= 1-em.getModel().getBeta(t,q,l); // cf. paper
	    }}}
      Rcpp::NumericVector gammadims(3);
      gammadims[0] = T; gammadims[1] = Q; gammadims[2] = Q; 
      Rcpp::Dimension d2(gammadims);                // get the dim object
      Rcpp::NumericVector gamma(d2);             // create vec. with correct dims
      for(int t=0;t<T;t++){
        for(int q=0;q<Q;q++){
          for(int l=0;l<Q;l++){
            gamma[l*(Q*T)+q*T+t]= em.getModel().getZTPlam(t,q,l);
	    }}}
      
      std::vector<int> distdims = em.getModel().getDistdims();
      Rcpp::List varphi(S); 
      for (int s=0;s<S;s++){
        Rcpp::NumericVector varphisdims(3);
        varphisdims[0]=T; varphisdims[1]=Q; varphisdims[2]=distdims[s];
        Rcpp::Dimension d(varphisdims);
        Rcpp::NumericVector varphis(d);
        for(int t=0;t<T;t++){
          for(int q=0;q<Q;q++){
            const std::vector<double> & temp = em.getModel().getVarphi(t,q,s);
            for(int ds=0;ds<distdims[s];ds++){
              varphis[ds*(Q*T)+q*T+t]=temp[ds];
            }
          }
        }
        varphi[s]=varphis;
      }

      Rcpp::NumericVector fintaumdims(3);
      fintaumdims[0] = T; 
      fintaumdims[1] = N; 
      fintaumdims[2] = Q;
      Rcpp::Dimension fintaud(fintaumdims); // get the dim object
      Rcpp::NumericVector fintaums(fintaud); // create vec. with correct dims
      if (ret_marginals){
        // return estimates of marginals
        for(int t=0;t<T;t++){
          for(int i=0; i<N; i++){
            for(int q=0;q<Q;q++){
              fintaums[q*(N*T)+i*T+t]=em.getModel().getfinTaum(t,i,q);
            }
          }
        }
      } 

      double lkl = em.getModel().modelselectionLoglikelihood(Y, X);

      
      deallocate3D<int>(Y,T,N,Q);
      deallocate3D<std::vector<double>>(X,T,N,S);
      
      if (!ret_marginals){
        return List::create(Rcpp::Named("trans") = trans,
			  Rcpp::Named("membership") = membership,
			  Rcpp::Named("beta") = beta,
			  Rcpp::Named("gamma") = gamma,
			  Rcpp::Named("loglikelihood") = lkl,
			  Rcpp::Named("iter") = nbiteff,
			  Rcpp::Named("directed") = isdirected,
			  Rcpp::Named("self.loop") = withselfloop,
        Rcpp::Named("varphi") = varphi);
      } else {
        return List::create(Rcpp::Named("trans") = trans,
			  Rcpp::Named("membership") = membership,
			  Rcpp::Named("beta") = beta,
			  Rcpp::Named("gamma") = gamma,
			  Rcpp::Named("loglikelihood") = lkl,
			  Rcpp::Named("iter") = nbiteff,
			  Rcpp::Named("directed") = isdirected,
			  Rcpp::Named("self.loop") = withselfloop,
        Rcpp::Named("varphi") = varphi,
        Rcpp::Named("fin.taum") = fintaums);
      }
    } else if (edgetype=="DCpoisson"){
      EM<DynDCSBMPoisson,int,std::vector<double>> em(T,N,Q,S,present,metapresent,metatypes,metadims,metatuning,isdirected,withselfloop);
      int*** Y;
      allocate3D<int>(Y,T,N,N);
      int p=0;
      for(int j=0; j<N; j++){
        for(int i=0; i<N; i++){
          for(int t=0; t<T; t++){
            Y[t][i][j] = int(Yasvector[p]);
            p++;
          }
        }
      }
      
      std::vector<double>*** X;
      allocate3Dvectors<double>(X,T,N,S,metadims);    
      for (int spos=0; spos<S; spos++){
        Rcpp::NumericVector Xsvector = Xaslist[spos];
        int p=0;
        for (int ds=0;ds<metadims[spos];ds++){
          for (int i=0;i<N;i++){
            for (int t=0;t<T;t++){
              X[t][i][spos][ds]=Xsvector[p];
              p++;
              if (metapresent[spos][t][i]!=0){
                for (const auto& x : X[t][i][spos]){
                  if (std::isnan(x)){
                    throw std::invalid_argument("Loading nan where metadata should be present: (t,i,s) = (" + std::to_string(t) +","+std::to_string(i) +","+std::to_string(spos)+") and p is "+std::to_string(p));
                  }
                }
              }
            }
          }
        }
      }
      // Rcpp::Rcout << "Successfully loaded metadata, here there are no nans when metapresent..." << "\n";
      
      em.initialize(as<vector<int> >(clustering),Y,X,frozen);
      int nbiteff = em.run(Y,X,nbit,10,frozen);
      NumericMatrix trans(Q,Q);
      for(int q=0;q<Q;q++) for(int l=0;l<Q;l++) trans[l+q*Q] = em.getModel().getTrans(q,l);
      IntegerMatrix membership(N,T);
      for(int t=0;t<T;t++){
        std::vector<int> groups = em.getModel().getGroupsByMAP(t);
        for(int i=0;i<N;i++) membership[i+t*N] = groups[i]+1;
      }
      
      Rcpp::NumericVector gammadims(3);
      gammadims[0] = T; gammadims[1] = Q; gammadims[2] = Q; 
      Rcpp::Dimension d2(gammadims);                // get the dim object
      Rcpp::NumericVector gamma(d2);             // create vec. with correct dims
      for(int t=0;t<T;t++){
        for(int q=0;q<Q;q++){
          for(int l=0;l<Q;l++){
            gamma[l*(Q*T)+q*T+t]= em.getModel().getDClam(t,q,l);
	    }}}
      
      std::vector<int> distdims = em.getModel().getDistdims();
      Rcpp::List varphi(S); 
      for (int s=0;s<S;s++){
        Rcpp::NumericVector varphisdims(3);
        varphisdims[0]=T; varphisdims[1]=Q; varphisdims[2]=distdims[s];
        Rcpp::Dimension d(varphisdims);
        Rcpp::NumericVector varphis(d);
        for(int t=0;t<T;t++){
          for(int q=0;q<Q;q++){
            const std::vector<double> & temp = em.getModel().getVarphi(t,q,s);
            for(int ds=0;ds<distdims[s];ds++){
              varphis[ds*(Q*T)+q*T+t]=temp[ds];
            }
          }
        }
        varphi[s]=varphis;
      }

      Rcpp::NumericVector fintaumdims(3);
      fintaumdims[0] = T; 
      fintaumdims[1] = N; 
      fintaumdims[2] = Q;
      Rcpp::Dimension fintaud(fintaumdims); // get the dim object
      Rcpp::NumericVector fintaums(fintaud); // create vec. with correct dims
      if (ret_marginals){
        // return estimates of marginals
        for(int t=0;t<T;t++){
          for(int i=0; i<N; i++){
            for(int q=0;q<Q;q++){
              fintaums[q*(N*T)+i*T+t]=em.getModel().getfinTaum(t,i,q);
            }
          }
        }
      } 

      double lkl = em.getModel().modelselectionLoglikelihood(Y, X);

      
      deallocate3D<int>(Y,T,N,Q);
      deallocate3D<std::vector<double>>(X,T,N,S);
      
      if (!ret_marginals){
        return List::create(Rcpp::Named("trans") = trans,
			  Rcpp::Named("membership") = membership,
			  Rcpp::Named("gamma") = gamma,
			  Rcpp::Named("loglikelihood") = lkl,
			  Rcpp::Named("iter") = nbiteff,
			  Rcpp::Named("directed") = isdirected,
			  Rcpp::Named("self.loop") = withselfloop,
        Rcpp::Named("varphi") = varphi);
      } else {
        return List::create(Rcpp::Named("trans") = trans,
			  Rcpp::Named("membership") = membership,
			  Rcpp::Named("gamma") = gamma,
			  Rcpp::Named("loglikelihood") = lkl,
			  Rcpp::Named("iter") = nbiteff,
			  Rcpp::Named("directed") = isdirected,
			  Rcpp::Named("self.loop") = withselfloop,
        Rcpp::Named("varphi") = varphi,
        Rcpp::Named("fin.taum") = fintaums);
      }
    } else {
      throw std::invalid_argument("Unknown edgetype passed");
>>>>>>> dynsbmmeta-cp/master
    }
  }
}
  
