/*
    This file is part of dynsbmmeta.

    dynsbmmeta is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    dynsbmmeta is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with dynsbmmeta.  If not, see <http://www.gnu.org/licenses/>
 */
#ifndef DYNSBM_EM_H
#define DYNSBM_EM_H
#include<DynSBM.h>
#include<DynDCSBMPoisson.h>
#include<iostream>
namespace dynsbm{
  template<class TDynSBM, typename Ytype, typename Xtype> // templates with type of DynSBM (Binary, Discrete, Gaussian, Poisson, DCPoisson), type of Y (int or double), and type of X (either int (categorical), double (continuous), or std::vector (cont vector and/or word embeddings for topics))
  class EM{
  private:
    TDynSBM _model;
  public:
    EM(int T, int N, int Q, int S, const Rcpp::IntegerMatrix& present, const std::vector<std::vector<std::vector<int>>>& metapresent, const std::vector<std::string> & metatypes, const std::vector<int> & metadims, const std::vector<double> & metatuning, bool isdirected = false, bool withselfloop = false)
      : _model(T,N,Q,S,present,metapresent,metatypes,metadims,metatuning,isdirected,withselfloop) {}
    ~EM(){};
    const TDynSBM& getModel() const{
      return _model;
    }
	
    void initialize(const std::vector<int>& clustering, Ytype*** const Y, Xtype*** const X, bool frozen=false);

    int run(Ytype*** const Y, Xtype*** const X, int nbit, int nbitFP, bool frozen);
	
  };
  
  // partial specialisation for DC initialisation
  template<typename Ytype, typename Xtype>
  class EM<DynDCSBMPoisson,Ytype,Xtype>{
  private:
    DynDCSBMPoisson _model;
  public:
    EM(int T, int N, int Q, int S, const Rcpp::IntegerMatrix& present, const std::vector<std::vector<std::vector<int>>>& metapresent, const std::vector<std::string> & metatypes, const std::vector<int> & metadims, const std::vector<double> & metatuning, bool isdirected = false, bool withselfloop = false)
      : _model(T,N,Q,S,present,metapresent,metatypes,metadims,metatuning,isdirected,withselfloop) {}
    ~EM(){};
    const DynDCSBMPoisson& getModel() const{
      return _model;
    }
	
    void initialize(const std::vector<int>& clustering, Ytype*** const Y, Xtype*** const X, bool frozen=false);

    int run(Ytype*** const Y, Xtype*** const X, int nbit, int nbitFP, bool frozen);
  };


  template<class TDynSBM, typename Ytype, typename Xtype>
  void EM<TDynSBM,Ytype,Xtype>::initialize(const std::vector<int>& clustering, Ytype*** const Y, Xtype*** const X, bool frozen){ 
	_model.initTau(clustering);
	if(frozen)
	  _model.updateFrozenTheta(Y);
	else
	  _model.updateTheta(Y);	
	  _model.initNotinformativeStationary();
	  _model.initNotinformativeTrans();
	  _model.updateVarphi(X);
  }

  template<class TDynSBM, typename Ytype, typename Xtype>
  int EM<TDynSBM,Ytype,Xtype>::run(Ytype*** const Y, Xtype*** const X, int nbit, int nbitFP, bool frozen){
      double prevlogl = _model.completedLoglikelihood(Y, X);
      //---- estimation
      int it = 0, nbiteff = 0;
      while(it<nbit){
		int itfp = 0;
		double prevloglfp = prevlogl;
		while (itfp<nbitFP){
			_model.updateTau(Y, X);
			if (itfp%3==0){ // saving time
				double newloglfp = _model.completedLoglikelihood(Y, X);
				if(fabs((prevloglfp-newloglfp)/prevloglfp)<1e-4){
				itfp = nbitFP;
				} else{
				prevloglfp = newloglfp;
				itfp = itfp+1;
				}
			} else
				itfp = itfp+1;
#ifdef DEBUG
		std::cerr<<"After EStep: "<<_model.completedLoglikelihood(Y, X)<<std::endl;
#endif
		}
		_model.updateTrans();
#ifdef DEBUG
		std::cerr<<"After MStep on trans: "<<_model.completedLoglikelihood(Y, X)<<std::endl;
#endif
		_model.updateStationary();
		if(frozen)
			_model.updateFrozenTheta(Y);
		else
			_model.updateTheta(Y);	  
#ifdef DEBUG
		std::cerr<<"After MStep on theta: "<<_model.completedLoglikelihood(Y, X)<<std::endl;
#endif
		_model.updateVarphi(X);
#ifdef DEBUG
		std::cerr<<"After MStep on varphi: "<<_model.completedLoglikelihood(Y, X)<<std::endl;
#endif
		double newlogl = _model.completedLoglikelihood(Y, X);
#ifdef DEBUG
		std::cerr<<"Testing the likelihood decrease: "<<prevlogl<<" -> "<<newlogl<<std::endl;
#endif
		nbiteff++;
		if(fabs((prevlogl-newlogl)/prevlogl)<1e-4){
#ifdef DEBUG
	  		std::cerr<<"Stopping: criteria is reached"<<std::endl;
#endif
	  		it=nbit;
		}
		if(prevlogl>newlogl){
#ifdef DEBUG
	 		std::cerr<<"Stopping: increasing logl"<<std::endl;
#endif
			it=nbit;
		}
		prevlogl = newlogl;
		it = it+1;
      }
      return(nbiteff);
    }
  
  template<typename Ytype, typename Xtype>
  void EM<DynDCSBMPoisson, Ytype, Xtype>::initialize(const std::vector<int>& clustering, Ytype*** const Y, Xtype*** const X, bool frozen){ 
      _model.initTau(clustering);
	  _model.initDegs(Y);
      if(frozen)
		_model.updateFrozenTheta(Y);
      else
		_model.updateTheta(Y);	
		_model.initNotinformativeStationary();
		_model.initNotinformativeTrans();
		_model.updateVarphi(X);
  }
  
  template<typename Ytype, typename Xtype>
  int EM<DynDCSBMPoisson,Ytype,Xtype>::run(Ytype*** const Y, Xtype*** const X, int nbit, int nbitFP, bool frozen){
      double prevlogl = _model.completedLoglikelihood(Y, X);
      //---- estimation
      int it = 0, nbiteff = 0;
      while(it<nbit){
		int itfp = 0;
		double prevloglfp = prevlogl;
		while (itfp<nbitFP){
			_model.updateTau(Y, X);
			if (itfp%3==0){ // saving time
				double newloglfp = _model.completedLoglikelihood(Y, X);
				if(fabs((prevloglfp-newloglfp)/prevloglfp)<1e-4){
				itfp = nbitFP;
				} else{
				prevloglfp = newloglfp;
				itfp = itfp+1;
				}
			} else
				itfp = itfp+1;
#ifdef DEBUG
		std::cerr<<"After EStep: "<<_model.completedLoglikelihood(Y, X)<<std::endl;
#endif
		}
		_model.updateTrans();
#ifdef DEBUG
		std::cerr<<"After MStep on trans: "<<_model.completedLoglikelihood(Y, X)<<std::endl;
#endif
		_model.updateStationary();
		if(frozen)
			_model.updateFrozenTheta(Y);
		else
			_model.updateTheta(Y);	  
#ifdef DEBUG
		std::cerr<<"After MStep on theta: "<<_model.completedLoglikelihood(Y, X)<<std::endl;
#endif
		_model.updateVarphi(X);
#ifdef DEBUG
		std::cerr<<"After MStep on varphi: "<<_model.completedLoglikelihood(Y, X)<<std::endl;
#endif
		double newlogl = _model.completedLoglikelihood(Y, X);
#ifdef DEBUG
		std::cerr<<"Testing the likelihood decrease: "<<prevlogl<<" -> "<<newlogl<<std::endl;
#endif
		nbiteff++;
		if(fabs((prevlogl-newlogl)/prevlogl)<1e-4){
#ifdef DEBUG
	  		std::cerr<<"Stopping: criteria is reached"<<std::endl;
#endif
	  		it=nbit;
		}
		if(prevlogl>newlogl){
#ifdef DEBUG
	 		std::cerr<<"Stopping: increasing logl"<<std::endl;
#endif
			it=nbit;
		}
		prevlogl = newlogl;
		it = it+1;
      }
      return(nbiteff);
    }
}
#endif
