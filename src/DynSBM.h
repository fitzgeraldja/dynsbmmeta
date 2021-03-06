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
    along with dynsbm.  If not, see <http://www.gnu.org/licenses/>
 */
#ifndef DYNSBM_DYNSBM_H
#define DYNSBM_DYNSBM_H
#include <Rcpp.h>
#include<vector>
//#include<math.h>
#include<tools.h>
#include<limits>
#include<iostream>
#include<stdexcept>

namespace dynsbm{
  template<typename Ytype, typename Xtype> 
  class DynSBM{
  protected:
    int _t; // time steps
    int _n; // number of nodes
    int _q; // number of groups
    int _s; // number of types of metadata (or total dims if allowing vectors)
    const std::vector<std::string> & _metatypes; //names of metadata distributions
    const std::vector<std::string> accepted_dists={"independent bernoulli","categorical","poisson","normal","negative binomial"}; // TO DO: code up function to check input metadists against these - just use str_contains
    bool _isdirected;
    bool _withselfloop;
    const Rcpp::IntegerMatrix & _present;
    bool ispresent(int t, int i) const{
      return(_present(i,t)!=0);
    }
    bool isabsent(int t, int i) const{
      return(_present(i,t)==0);
    }
    // NB need to introduce _metapresent for metadata for cases where have metadata but no edges
    const std::vector<std::vector<std::vector<int>>> & _metapresent;
    bool ismetapresent(int t, int i, int s) const{
      return(_metapresent[s][t][i]!=0); 
    }
    bool ismetaabsent(int t, int i, int s) const{
      return(_metapresent[s][t][i]==0); 
    }
    const std::vector<int> & _metadims;

    const std::vector<double> & _metatuning;
    
    // Markov chain
    double* _stationary;
    double** _trans;
    
    // variational EM
    double** _tau1;
    double**** _taut;
    double*** _taum;
    void correctTau1();
    void correctTaut();
    void correctTaum();
    void updateTauMarginal();
    double tauMarginal(int t, int i, int q) const; // proba of q iif present(t,i)==true
    double tauArrival(int t, int i, int q) const{ // proba of q when absent before t>0
      int tstorage = t-1;
      return(_taut[tstorage][i][0][q]);
    }

    // edge distribution
    double*** _betaql; // trick: store log(betaql) - NB this numerical stability trick (incl subtraction within exponential of max value before readding) discussed in S2.5 of Bleis review of VI for statisticians
    double*** _1minusbetaql; // trick: store log(1-betaql)
    void correctBeta();
    virtual double logDensity(int t, int q, int l, Ytype y) const = 0; // this logDensity refers to SBM probs, not full density
    template<class TAddEventFunctor> 
    void updateThetaCore(Ytype*** const Y, TAddEventFunctor& addEventFunctor);
    template<class TAddEventFunctor> 
    void updateFrozenThetaCore(Ytype*** const Y, TAddEventFunctor& addEventFunctor);
    
    // metadata distribution
    // double*** _varphit; // T x Q x S, trick: store log(varphit) - can't use as unknown number of parameters without knowing distributions
    std::vector<double>*** _varphi; // 4d array: T x Q x S x number of params (can be different for each row) - note for matrices of parameters would have to flatten (e.g. Gaussian)
    std::vector<int> _distdims;
    double logMetadensity(int t, int q, int s, Xtype x) const; // llkl of metadata x of type s for group q at time t
    void correctVarphi();
  
    
  public:
    DynSBM(int T, int N, int Q, int S, const Rcpp::IntegerMatrix & present, const std::vector<std::vector<std::vector<int>>> & metapresent, const std::vector<std::string> & metatypes, const std::vector<int> & metadims, const std::vector<double> & metatuning, bool isdirected = false, bool withselfloop = false);
    ~DynSBM();
    // Markov chain
    double getfinTaum(int t, int i, int q) const{
      return(tauMarginal(t,i,q));
    }
    double getTrans(int q, int l) const{
      return(_trans[q][l]);
    }
    double getBeta(int t, int q, int l) const{
      // optimization trick: stored in log
      return(exp(_betaql[t][q][l]));
    }
    std::vector<double> getVarphi(int t, int q, int s) const{
      // optimisation trick: stored in log
      return(_varphi[t][q][s]);
    }
    int getDistdimsinit(const std::string & distname, int metadim = 1){
      if (distname=="poisson"){
        return 1;
      }else if (distname=="negative binomial"){
        return 2;
      }else if (str_contains(distname,"categorical")){
        std::string delimiter = "x ";
        int l = std::stoi(distname.substr(distname.find(delimiter)+2,distname.length()));
        return l;
      }else if (str_contains(distname,"independent bernoulli")){
        std::string delimiter = "x ";
        int l = std::stoi(distname.substr(distname.find(delimiter)+2,distname.length()));
        return l;
      }else if (str_contains(distname,"normal")){
        if (str_contains(distname,"independent")){
          return metadim*2;
        }else if (str_contains(distname,"shared")){
          return metadim+1;
        }else{
          return metadim*(1+metadim);
        }
      }else{
        throw std::invalid_argument("Unrecognised distribution passed:\nChoose from poisson, negative binomial,\ncategorical x L, independent bernoulli x L, (shared/independent) normal");
      }
    }
    std::vector<int> getDistdims() const{
      return(_distdims);
    }
    
    void updateTrans();
    void updateStationary();
    void initNotinformativeStationary();
    void initNotinformativeTrans();
    // variational EM
    void initTau(const std::vector<int>& clustering);
    void updateTau(Ytype*** const Y, Xtype*** const X);
    virtual void updateTheta(Ytype*** const Y) = 0;
    virtual void updateFrozenTheta(Ytype*** const Y) = 0;
    double completedLoglikelihood(Ytype*** const Y, Xtype*** const X ) const;

    // metadata
    void updateVarphi(Xtype*** const X);
    
    // model output
    std::vector<int> getGroupsByMAP(int t) const;
    double modelmetadataLoglikelihood(Xtype*** const X) const;
    double modelselectionLoglikelihood(Ytype*** const Y, Xtype*** const X) const;
    
  };
  
  template<typename Ytype, typename Xtype> 
  DynSBM<Ytype,Xtype>::DynSBM(int T, int N, int Q, int S, const Rcpp::IntegerMatrix & present, const std::vector<std::vector<std::vector<int>>> & metapresent,const std::vector<std::string> & metatypes,  const std::vector<int> & metadims, const std::vector<double> & metatuning, bool isdirected, bool withselfloop)
    : _t(T),_n(N),_q(Q),_s(S),_isdirected(isdirected),_withselfloop(withselfloop),_present(present),_metapresent(metapresent),_metatypes(metatypes),_metadims(metadims),_metatuning(metatuning){
    _stationary = new double[_q];
    allocate2D<double>(_trans,_q,_q);
    allocate2D<double>(_tau1,_n,_q);
    allocate4D<double>(_taut,_t-1,_n,_q,_q);
    allocate3D<double>(_taum,_t,_n,_q);
    allocate3D<double>(_betaql,_t,_q,_q);
    allocate3D<double>(_1minusbetaql,_t,_q,_q);
    _distdims.resize(_s);
    for (int i=0;i<_s;i++){
      _distdims[i]=getDistdimsinit(_metatypes[i],_metadims[i]);
    }
    allocate3Dvectors<double>(_varphi,_t,_q,_s,_distdims); 
  }
  
  template<typename Ytype, typename Xtype> 
  DynSBM<Ytype, Xtype>::~DynSBM(){
    delete[] _stationary;
    deallocate2D<double>(_trans,_q,_q);
    deallocate2D<double>(_tau1,_n,_q);
    deallocate4D<double>(_taut,_t-1,_n,_q,_q);
    deallocate3D<double>(_taum,_t,_n,_q);
    deallocate3D<double>(_betaql,_t,_q,_q);
    deallocate3D<double>(_1minusbetaql,_t,_q,_q);
    deallocate3D<std::vector<double>>(_varphi,_t,_q,_s);
  }
  
  template<typename Ytype, typename Xtype> 
  void DynSBM<Ytype, Xtype>::updateTrans(){
    if (_q==1){
      _trans[0][0] = 1.;
      return;
    }
    double** denom = new double*[_q]; // denominator of transition estim
    for(int q=0;q<_q;q++) denom[q] = new double[_q];
    for(int q=0;q<_q;q++) 
      for(int l=0;l<_q;l++){
	      _trans[q][l] = 0.; denom[q][l] = 0.;
      }
    for(int t=1;t<_t;t++){
      int tstorage = t-1;
      for(int i=0;i<_n;i++){
	      if (ispresent(t-1,i) && ispresent(t,i)){
	        for(int q=0;q<_q;q++){
	          for(int qprime=0;qprime<_q;qprime++){
              _trans[q][qprime] += _taut[tstorage][i][q][qprime] * tauMarginal(t-1,i,q);
              denom[q][qprime] += tauMarginal(t-1,i,q);
	          }
	        }
	      }
      }
    }
    // numerical issue : avoid too small value for trans
    for(int q=0;q<_q;q++){
      double sumtrans=0.;
      for(int qprime=0;qprime<_q;qprime++){
        _trans[q][qprime] = _trans[q][qprime] / denom[q][qprime];
        sumtrans += _trans[q][qprime];
      }
      if (sumtrans>0) for(int qprime=0;qprime<_q;qprime++) _trans[q][qprime] = _trans[q][qprime]/sumtrans;
      for(int qprime=0;qprime<_q;qprime++){
        if(_trans[q][qprime]<precision){
          _trans[q][qprime] = precision;
        }
      }
      sumtrans=0;
      for(int qprime=0;qprime<_q;qprime++) sumtrans += _trans[q][qprime];
      if (sumtrans>0) for(int qprime=0;qprime<_q;qprime++) _trans[q][qprime] = _trans[q][qprime]/sumtrans;
    }
    for(int q=0;q<_q;q++) delete[] denom[q];
    delete[] denom;
  }
  
  template<typename Ytype, typename Xtype> 
  void DynSBM<Ytype, Xtype>::updateStationary(){
    if (_q==1){
      _stationary[0] = 1.;
      return;
    }
    for(int q=0;q<_q;q++) _stationary[q]=0.;
    double sumst=0;
    for(int q=0;q<_q;q++){
      for(int t=0;t<_t;t++){
        for(int i=0;i<_n;i++){
          if (ispresent(t,i))
            _stationary[q] += tauMarginal(t,i,q);
        }
      }
      if(_stationary[q]<precision){
	      _stationary[q] = precision;
      }
      sumst += _stationary[q];
    }
    for(int q=0;q<_q;q++) _stationary[q] = _stationary[q]/sumst;
  }
  
  template<typename Ytype, typename Xtype> 
  void DynSBM<Ytype, Xtype>::initNotinformativeStationary(){
    for(int q=0;q<_q;q++) _stationary[q] = 1./_q;
  }
  
  template<typename Ytype, typename Xtype> 
  void DynSBM<Ytype, Xtype>::initNotinformativeTrans(){
    for(int q=0;q<_q;q++) for(int l=0;l<_q;l++) _trans[q][l] = 1./_q;
  }
  
  template<typename Ytype, typename Xtype>
  void DynSBM<Ytype, Xtype>::updateTau(Ytype*** const Y, Xtype*** const X){
    if(_q==1) return;
    double** newtau1;
    allocate2D<double>(newtau1,_n,_q);
    for(int i=0;i<_n;i++) 
      for(int q=0;q<_q;q++) 
	      newtau1[i][q] = 0.;
    double**** newtaut;
    allocate4D<double>(newtaut,_t-1,_n,_q,_q); //is this not just doing what the loops below do...
    for(int t=0;t<_t-1;t++) 
      for(int i=0;i<_n;i++) 
        for(int q=0;q<_q;q++) 
          for(int l=0;l<_q;l++) 
            newtaut[t][i][q][l] = 0;  
    // t=1
#pragma omp parallel for
    for(int i=0;i<_n;i++){
      if (ispresent(0,i)){
        double maxlogtau1i = -std::numeric_limits<double>::max();
        std::vector<double> logtau1i(_q,0.); // initialise length _q vec w zeroes
        for(int q=0;q<_q;q++){
          double logp = 0.;
          for(int j=0;j<i;j++){
            if (ispresent(0,j)){
              for(int l=0;l<_q;l++){
                if (!std::isnan(Y[0][i][j]))
                  logp += _tau1[j][l]*logDensity(0,q,l,Y[0][i][j]);
                  if (_isdirected) logp += _tau1[j][l]*logDensity(0,l,q,Y[0][j][i]);
              }
            }
          }
          if((_withselfloop)&(!std::isnan(Y[0][i][i]))) logp += logDensity(0,q,q,Y[0][i][i]); // missing tau1? must be as selfloop
          for(int j=i+1;j<_n;j++){
            if (ispresent(0,j)){
              for(int l=0;l<_q;l++){
                if (!std::isnan(Y[0][i][j]))
                  logp += _tau1[j][l]*logDensity(0,q,l,Y[0][i][j]);
                  if (_isdirected) logp += _tau1[j][l]*logDensity(0,l,q,Y[0][j][i]);
              }
            }
          }
          // modified to include metadata info
          double logm = 0.;
          for (int s=0;s<_s;s++){
            if (ismetapresent(0,i,s)){
              // NB would need to check size _metatuning here in general, then use below if not 1
              logm+=_metatuning[0]*logMetadensity(0,q,s,X[0][i][s]);
              // if (std::isnan(logm)){
              //   std::string Xasstring = "[";
              //   for (const auto & x : X[0][i][s]){
              //     Xasstring += std::to_string(x);
              //     Xasstring += ", ";
              //   }
              //   Xasstring += "]";
              //   throw std::invalid_argument("Meta logdensity has become nan - must have fault in meta.present.\ni is "+std::to_string(i)+", s is "+std::to_string(s)+", X is "+Xasstring);
              // }

              // logm+=_metatuning[q]*(logMetadensity(0,q,s,X[0][i][s])+log(_metatuning[q]))+(1-_metatuning[q])*(log(1-_metatuning[q])+logGlobmetadensity(0,s,X[0][i][s]));
            }
          }
          
          // _stationary refers to alpha
          logtau1i[q] = logp + log(_stationary[q]) + logm;
          if (logtau1i[q]>maxlogtau1i) maxlogtau1i = logtau1i[q];
        }
        // numerical issue : normalization
        std::vector<double> tau1i(_q,0);
        double sumtau1i = 0.;
        for(int q=0;q<_q;q++){
          tau1i[q] = exp(logtau1i[q]-maxlogtau1i);
          sumtau1i = sumtau1i+tau1i[q];
        }
        for(int q=0;q<_q;q++) newtau1[i][q] = tau1i[q]/sumtau1i;
      }
    }
    // t>1
    for(int t=1;t<_t;t++){
      int tstorage = t-1;
#pragma omp parallel for
      for(int i=0;i<_n;i++){
        if (ispresent(t,i)){
          std::vector<double> logps(_q,0.);
          for(int qprime=0;qprime<_q;qprime++){
            double logp = 0.;
            // break into j<i and j>i so can deal with self loops separately
            for(int j=0;j<i;j++){
              if (ispresent(t,j)){
                for(int l=0;l<_q;l++){
                  if (!std::isnan(Y[t][i][j]))
                    logp += tauMarginal(t,j,l)*logDensity(t,qprime,l,Y[t][i][j]);
                    if (_isdirected) logp += tauMarginal(t,j,l)*logDensity(t,l,qprime,Y[t][j][i]);
                }
              }
            }
	          if((_withselfloop)&(!std::isnan(Y[t][i][i]))) logp += logDensity(t,qprime,qprime,Y[t][i][i]);
            for(int j=i+1;j<_n;j++){
              if (ispresent(t,j)){
                for(int l=0;l<_q;l++){
                  if (!std::isnan(Y[t][i][j]))
                    logp += tauMarginal(t,j,l)*logDensity(t,qprime,l,Y[t][i][j]);
                    if (_isdirected) logp += tauMarginal(t,j,l)*logDensity(t,l,qprime,Y[t][j][i]);
                }
              }
            }
            logps[qprime] = logp; // logps is the SBM likelihood for all q'
          }
          for(int q=0;q<(ispresent(t-1,i)?_q:1);q++){ // only q=0 if absent at t-1 (see tauArrival method)
            double maxlogtauti = -std::numeric_limits<double>::max();
            std::vector<double> logtauti(_q,0.);
            if (ispresent(t-1,i)){
              for(int qprime=0;qprime<_q;qprime++){
                // modified for metadata
                double logm = 0.;
                for(int s=0;s<_s;s++){
                  if (ismetapresent(t,i,s)){
                    logm += _metatuning[0]*logMetadensity(t,qprime,s,X[t][i][s]);
                    // NB would need to check size _metatuning here in general, then use below if not 1
                    // logm+=_metatuning[qprime]*(logMetadensity(t,qprime,s,X[t][i][s])+log(_metatuning[qprime]))+(1-_metatuning[qprime])*(log(1-_metatuning[qprime])+logGlobmetadensity(t,s,X[t][i][s]));
                  }
                }
                
                logtauti[qprime] = logps[qprime] + log(_trans[q][qprime]) + logm;
                if (logtauti[qprime]>maxlogtauti) maxlogtauti = logtauti[qprime];
              }
            }else{
              // no need to modify for missing (meta)data - assume if no info then no modification
              // might want to change to just have non-informative prior over metadata in this case
              for(int qprime=0;qprime<_q;qprime++){
                logtauti[qprime] = logps[qprime] + log(_stationary[qprime]);
                if (logtauti[qprime]>maxlogtauti) maxlogtauti = logtauti[qprime];
              }
            }
            // numerical issue : normalization
            std::vector<double> tauti(_q,0);
            double sumtauti = 0.;
            for(int qprime=0;qprime<_q;qprime++){
              tauti[qprime] = exp(logtauti[qprime]-maxlogtauti);
              sumtauti = sumtauti+tauti[qprime];
            }
            for(int qprime=0;qprime<_q;qprime++){
              tauti[qprime] = tauti[qprime]/sumtauti;
              newtaut[tstorage][i][q][qprime] = tauti[qprime];
            }
          }
        }
      }
    }
    // switching tau for old values at iteration (k) to new values at iteration (k+1)
    for(int i=0;i<_n;i++){
      for(int q=0;q<_q;q++){ 
	      _tau1[i][q] = newtau1[i][q];
      }
    }
    for(int t=0;t<_t-1;t++){
      for(int i=0;i<_n;i++){
        for(int q=0;q<_q;q++){ 
          for(int l=0;l<_q;l++){ 
            _taut[t][i][q][l] = newtaut[t][i][q][l];
          }
        }
      }
    }
    deallocate2D<double>(newtau1,_n,_q);
    deallocate4D<double>(newtaut,_t-1,_n,_q,_q);
    correctTau1();
    correctTaut();
    updateTauMarginal();
  }
    
  template<typename Ytype, typename Xtype>
  template<class TAddEventFunctor> 
  void DynSBM<Ytype, Xtype>::updateThetaCore(Ytype*** const Y, TAddEventFunctor& addEventFunctor){// M-step
    for(int t=0;t<_t;t++){
      for(int q=0;q<_q;q++){
        for(int l=0;l<_q;l++){
          _betaql[t][q][l] = 0; // probability of a zero value
        }
      }
    }
    // note the constraint betaql[t,q,q] = betaql[,q,q]
    double*** betasumql;
    allocate3D<double>(betasumql,_t,_q,_q);
    // mu/beta
    for(int t=0;t<_t;t++){
      for(int i=0;i<_n;i++){
        if (ispresent(t,i)){
          // j<i
          for(int j=0;j<i;j++){
            if (ispresent(t,j)){
              for(int q=0;q<_q;q++){
                for(int l=0;l<q;l++){
                  if(_isdirected){
                    if (!std::isnan(Y[t][i][j])){		      
                      if(Y[t][i][j]>Ytype(0)){
                        addEventFunctor(tauMarginal(t,i,q)*tauMarginal(t,j,l), Y[t][i][j], t, q, l);
                        addEventFunctor(tauMarginal(t,i,l)*tauMarginal(t,j,q), Y[t][i][j], t, l, q);
                      }
                      else{
                        _betaql[t][q][l] += tauMarginal(t,i,q)*tauMarginal(t,j,l);
                        _betaql[t][l][q] += tauMarginal(t,i,l)*tauMarginal(t,j,q);
                      }
                      betasumql[t][q][l] += tauMarginal(t,i,q)*tauMarginal(t,j,l);
                      betasumql[t][l][q] += tauMarginal(t,i,l)*tauMarginal(t,j,q);
                    }
                    if (!std::isnan(Y[t][j][i])){
                      if(Y[t][j][i]>Ytype(0)){
                        addEventFunctor(tauMarginal(t,j,q)*tauMarginal(t,i,l), Y[t][j][i], t, q, l);
                        addEventFunctor(tauMarginal(t,j,l)*tauMarginal(t,i,q), Y[t][j][i], t, l, q);
                      } else{
                        _betaql[t][q][l] += tauMarginal(t,j,q)*tauMarginal(t,i,l);
                        _betaql[t][l][q] += tauMarginal(t,j,l)*tauMarginal(t,i,q);
                      }
                      betasumql[t][q][l] += tauMarginal(t,j,q)*tauMarginal(t,i,l);
                      betasumql[t][l][q] += tauMarginal(t,j,l)*tauMarginal(t,i,q);
                    }
                  } else{
                    if (!std::isnan(Y[t][i][j])){
                      if(Y[t][i][j]>Ytype(0)){
                        addEventFunctor(tauMarginal(t,i,q)*tauMarginal(t,j,l), Y[t][i][j], t, q, l);
                        addEventFunctor(tauMarginal(t,i,l)*tauMarginal(t,j,q), Y[t][i][j], t, q, l);
                      }
                      else{
                        _betaql[t][q][l] += tauMarginal(t,i,q)*tauMarginal(t,j,l) + tauMarginal(t,i,l)*tauMarginal(t,j,q);
                      }
                      betasumql[t][q][l] += tauMarginal(t,i,q)*tauMarginal(t,j,l) + tauMarginal(t,i,l)*tauMarginal(t,j,q);
                    }
                  }
                }
                // q==l
                if (!std::isnan(Y[t][i][j])){
                  if(Y[t][i][j]>Ytype(0))
                    addEventFunctor(tauMarginal(t,i,q)*tauMarginal(t,j,q), Y[t][i][j], 0, q, q);
                  else{
                    _betaql[0][q][q] += tauMarginal(t,i,q)*tauMarginal(t,j,q); // t=0 gathers all t when q=l
                  }
                }
                if(_isdirected){
                  if (!std::isnan(Y[t][j][i])){
                    if(Y[t][j][i]>Ytype(0)){
                      addEventFunctor(tauMarginal(t,i,q)*tauMarginal(t,j,q), Y[t][j][i], 0, q, q);
                    }else{
                      _betaql[0][q][q] += tauMarginal(t,i,q)*tauMarginal(t,j,q);
                    }
                  }
                }
                betasumql[0][q][q] += (1+int(_isdirected)) * tauMarginal(t,i,q)*tauMarginal(t,j,q);	
              }
            }
          }
          // j==i considered only if selfloop allowed
          if (_withselfloop){
            for(int q=0;q<_q;q++){
              if(Y[t][i][i]>Ytype(0)){
                addEventFunctor(tauMarginal(t,i,q), Y[t][i][i], 0, q, q);
              }else{
                _betaql[0][q][q] += tauMarginal(t,i,q); // t=0 gathers all t when q=l
              }
              betasumql[0][q][q] += tauMarginal(t,i,q);	    
            }	
          }
        }	
      }
    }
    for(int t=0;t<_t;t++){// symmetrize+normalize
      for(int q=(_isdirected?0:1);q<_q;q++){
        for(int l=0;l<q;l++){
          if (betasumql[t][q][l]>0){
            _betaql[t][q][l] = _betaql[t][q][l] / betasumql[t][q][l];
            if(!_isdirected) _betaql[t][l][q] = _betaql[t][q][l];
          }        	
        }
        if(_isdirected){
          for(int l=q+1;l<_q;l++){
            if (betasumql[t][q][l]>0){
              _betaql[t][q][l] = _betaql[t][q][l] / betasumql[t][q][l];	  
            }
          }
        }    
      }
    }
    for(int q=0;q<_q;q++){// symmetrize+normalize
      // note the constraint betaql[t,q,q] = betaql[,q,q]
      if (betasumql[0][q][q]>0){
        _betaql[0][q][q] = _betaql[0][q][q] / betasumql[0][q][q];
        for(int t=1;t<_t;t++){
          _betaql[t][q][q] = _betaql[0][q][q];
        }
      }
    }
    correctBeta();
    deallocate3D<double>(betasumql,_t,_q,_q);
  }

  /*
  template<typename Ytype>
  template<class TAddEventFunctor> 
  void DynSBM<Ytype>::updateFrozenThetaCore(Ytype*** const Y, TAddEventFunctor& addEventFunctor){// M-step
    ... SAME THING
    // symmetrize+normalize
    for(int q=_isdirected?0:1;q<_q;q++){
      for(int l=0;l<q;l++){
	double beta = 0., betasum = 0.;
	for(int t=0;t<_t;t++){
	  beta += _betaql[t][q][l];
	  betasum += betasumql[t][q][l];
	}
	if (betasum>0){	
	  for(int t=0;t<_t;t++){  
	    _betaql[t][q][l] = beta / betasum;
	    if(!_isdirected) _betaql[t][l][q] = _betaql[t][q][l];
	  }
	}
      }
      if(_isdirected)
	for(int l=q+1;l<_q;l++){
	  double beta = 0., betasum = 0.;
	  for(int t=0;t<_t;t++){
	    beta += _betaql[t][q][l];
	    betasum += betasumql[t][q][l];
	  }
	  if (betasum>0){	
	    for(int t=0;t<_t;t++){  
	      _betaql[t][q][l] = beta / betasum;
	    }
	  }	  
	}
    }
    
    for(int q=0;q<_q;q++){// symmetrize+normalize
      // note the constraint betaql[t,q,q] = betaql[,q,q]
      if (betasumql[0][q][q]>0)
	_betaql[0][q][q] = _betaql[0][q][q] / betasumql[0][q][q];
      for(int t=1;t<_t;t++)
	_betaql[t][q][q] = _betaql[0][q][q];
    }
    correctBeta();
    deallocate3D<double>(betasumql,_t,_q,_q);
  }
  */
  
  template<typename Ytype, typename Xtype>
  template<class TAddEventFunctor> 
  void DynSBM<Ytype, Xtype>::updateFrozenThetaCore(Ytype*** const Y, TAddEventFunctor& addEventFunctor){// M-step
    for(int t=0;t<_t;t++){
      for(int q=0;q<_q;q++){
        for(int l=0;l<_q;l++){
          _betaql[t][q][l] = 0; // probability of a zero value
        }
      }
    }
    // note the constraint betaql[t,q,q] = betaql[,q,q]
    double*** betasumql;
    allocate3D<double>(betasumql,_t,_q,_q);
    // mu/beta
    for(int t=0;t<_t;t++){
      for(int i=0;i<_n;i++){
        if (ispresent(t,i)){
          // j<i
          for(int j=0;j<i;j++){
            if (ispresent(t,j)){
              for(int q=0;q<_q;q++){
                for(int l=0;l<q;l++){
                  if(_isdirected){		      
                    if(Y[t][i][j]>Ytype(0)){
                      addEventFunctor(tauMarginal(t,i,q)*tauMarginal(t,j,l), Y[t][i][j], 0, q, l);
                      addEventFunctor(tauMarginal(t,i,l)*tauMarginal(t,j,q), Y[t][i][j], 0, l, q);
                    }
                    else{
                      _betaql[0][q][l] += tauMarginal(t,i,q)*tauMarginal(t,j,l);
                      _betaql[0][l][q] += tauMarginal(t,i,l)*tauMarginal(t,j,q);
                    }
                    betasumql[0][q][l] += tauMarginal(t,i,q)*tauMarginal(t,j,l);
                    betasumql[0][l][q] += tauMarginal(t,i,l)*tauMarginal(t,j,q);
                    if(Y[t][j][i]>Ytype(0)){
                      addEventFunctor(tauMarginal(t,j,q)*tauMarginal(t,i,l), Y[t][j][i], 0, q, l);
                      addEventFunctor(tauMarginal(t,j,l)*tauMarginal(t,i,q), Y[t][j][i], 0, l, q);
                    } else{
                      _betaql[0][q][l] += tauMarginal(t,j,q)*tauMarginal(t,i,l);
                      _betaql[0][l][q] += tauMarginal(t,j,l)*tauMarginal(t,i,q);
                    }
                    betasumql[0][q][l] += tauMarginal(t,j,q)*tauMarginal(t,i,l);
                    betasumql[0][l][q] += tauMarginal(t,j,l)*tauMarginal(t,i,q);
                  } else{
                    if(Y[t][i][j]>Ytype(0)){
                      addEventFunctor(tauMarginal(t,i,q)*tauMarginal(t,j,l), Y[t][i][j], 0, q, l);
                      addEventFunctor(tauMarginal(t,i,l)*tauMarginal(t,j,q), Y[t][i][j], 0, q, l);
                    }
                    else{
                      _betaql[0][q][l] += tauMarginal(t,i,q)*tauMarginal(t,j,l) + tauMarginal(t,i,l)*tauMarginal(t,j,q);
                    }
                    betasumql[0][q][l] += tauMarginal(t,i,q)*tauMarginal(t,j,l) + tauMarginal(t,i,l)*tauMarginal(t,j,q);
                  }
                }
                // q==l
                if(Y[t][i][j]>Ytype(0)){
                  addEventFunctor(tauMarginal(t,i,q)*tauMarginal(t,j,q), Y[t][i][j], 0, q, q);
                }else{
                  _betaql[0][q][q] += tauMarginal(t,i,q)*tauMarginal(t,j,q); // t=0 gathers all t when q=l
                }
                if(_isdirected){
                  if(Y[t][j][i]>Ytype(0)){
                    addEventFunctor(tauMarginal(t,i,q)*tauMarginal(t,j,q), Y[t][j][i], 0, q, q);
                  }else{
                    _betaql[0][q][q] += tauMarginal(t,i,q)*tauMarginal(t,j,q);
                  }
                }
                betasumql[0][q][q] += (1+int(_isdirected)) * tauMarginal(t,i,q)*tauMarginal(t,j,q);	
              }
            }
          }
          // j==i considered only if selfloop allowed
          if (_withselfloop){
            for(int q=0;q<_q;q++){
              if(Y[t][i][i]>Ytype(0)){
                addEventFunctor(tauMarginal(t,i,q), Y[t][i][i], 0, q, q);
              }else{
                _betaql[0][q][q] += tauMarginal(t,i,q); // t=0 gathers all t when q=l
              }
              betasumql[0][q][q] += tauMarginal(t,i,q);	    
            }
          }	
        }	
      }
    }
    // symmetrize+normalize
    for(int q=(_isdirected?0:1);q<_q;q++){
      for(int l=0;l<q;l++){
        if (betasumql[0][q][l]>0){
          _betaql[0][q][l] = _betaql[0][q][l] / betasumql[0][q][l];
          if(!_isdirected) _betaql[0][l][q] = _betaql[0][q][l];
        }
      }        	
      if(_isdirected){
        for(int l=q+1;l<_q;l++){
          if (betasumql[0][q][l]>0){
            _betaql[0][q][l] = _betaql[0][q][l] / betasumql[0][q][l];	      
          }
        }
      }
    }
    for(int q=0;q<_q;q++){// symmetrize+normalize
      if (betasumql[0][q][q]>0){
	      _betaql[0][q][q] = _betaql[0][q][q] / betasumql[0][q][q];
      }
    }
    // note the constraint betaql[t,q,l] = betaql[0,q,l]
    for(int t=1;t<_t;t++){
      for(int q=0;q<_q;q++){
        for(int l=0;l<_q;l++){
          _betaql[t][q][l] = _betaql[0][q][l];
        }
      }
    }
    correctBeta();
    deallocate3D<double>(betasumql,_t,_q,_q);
  }
  
  
  template<typename Ytype, typename Xtype>
  double DynSBM<Ytype, Xtype>::completedLoglikelihood(Ytype*** const Y, Xtype*** const X) const{ // including entropy term
    double J = 0.;
    //std::cerr.precision(8);
    // term 1
    for(int i=0;i<_n;i++){
      if (ispresent(0,i)){
        for(int q=0;q<_q;q++){
          J += _tau1[i][q] * (log(_stationary[q])-log(_tau1[i][q]));
        }
      }
    }
    //std::cerr<<"\nJ: "<<J<<" ";
    // term 2
#pragma omp parallel for reduction(+:J)
    for(int t=1;t<_t;t++){
      int tstorage = t-1;
      for(int i=0;i<_n;i++)
        if(ispresent(t,i)){
          if(ispresent(t-1,i))
            for(int q=0;q<_q;q++)
              for(int qprime=0;qprime<_q;qprime++)
                J += tauMarginal(t-1,i,q) * _taut[tstorage][i][q][qprime]
                  * (log(_trans[q][qprime])-log(_taut[tstorage][i][q][qprime]));
          else
            for(int q=0;q<_q;q++)
              J += tauArrival(t,i,q) * (log(_stationary[q])-log(tauArrival(t,i,q)));
	      }
    }
    //std::cerr<<J<<" ";
    // term 3
#pragma omp parallel for reduction(+:J)
    for(int t=0;t<_t;t++)
      for(int i=0;i<_n;i++)
        if (ispresent(t,i)){
          for(int j=0;j<i;j++)
            if (ispresent(t,j))
              for(int q=0;q<_q;q++){
                double taumtiq = tauMarginal(t,i,q);
                for(int l=0;l<_q;l++){
                  J += taumtiq*tauMarginal(t,j,l)*logDensity(t,q,l,Y[t][i][j]);
                  if (_isdirected) J += taumtiq*tauMarginal(t,j,l)*logDensity(t,l,q,Y[t][j][i]);
		            }
	            }
          if (_withselfloop){
            for(int q=0;q<_q;q++){
              J += tauMarginal(t,i,q)*logDensity(t,q,q,Y[t][i][i]);
	          }
	        }
	      }
    // term 4 - metadata
#pragma omp parallel for reduction(+:J)
    for(int t=0;t<_t;t++){
      for(int i=0;i<_n;i++){
        if (ispresent(t,i)){ // must check else tauMarginal(t,i,q) not defined as currently implemented
          for(int s=0;s<_s;s++){
            if (ismetapresent(t,i,s)){
              for(int q=0;q<_q;q++){
                J+=tauMarginal(t,i,q)*logMetadensity(t,q,s,X[t][i][s]);
              }
            }
          }
        }
      }
    }

    //std::cerr<<J<<std::endl;
    
    return(J);
  }
  
  // template<typename Ytype, typename Xtype>
  // double DynSBM<Ytype,Xtype>::modelmetadataLoglikelihood(Xtype*** const X) const{
  //   double METALKL = 0;
  //   for(int t=0;t<_t;t++){
  //     std::vector<int> groupst = getGroupsByMAP(t);
  //     for(int i=0;i<_n;i++){
  //       for(int s=0;s<_s;s++){
  //         if (ismetapresent(t,i,s)){
  //           METALKL += logMetadensity(t,groupst[i],s,X[t][i][s]);
  //         }
  //       }
  //     }
  //   }
  // }
 
  template<typename Ytype, typename Xtype>
  double DynSBM<Ytype,Xtype>::modelselectionLoglikelihood(Ytype*** const Y, Xtype*** const X) const{
    double LKL = 0;
    // term 1
    std::vector<int> groups1 = getGroupsByMAP(0);
    for(int i=0;i<_n;i++)
      if (ispresent(0,i))
	      LKL += log(_stationary[groups1[i]]);
    //std::cerr<<"\nLKL: "<<LKL<<" ";
    // term 2
    std::vector<int> groupstm1 = groups1;
    for(int t=1;t<_t;t++){
      std::vector<int> groupst = getGroupsByMAP(t);
      for(int i=0;i<_n;i++)
        if(ispresent(t,i)){
          if(ispresent(t-1,i)){
            try{
              // showing some nodes that are present still somehow have group -1
              // // only possible if all _tau1[i][q]=0 (shouldn't be possible given correctTau1), and/or all tauMarginal(t,i,q) = 0
              // Rcpp::Rcout << "t " << t << " i " << i << "\n";
              // Rcpp::Rcout << "groupstm1[i] " << groupstm1.at(i) << "\n";
              // Rcpp::Rcout << "groupst[i] " << groupst.at(i) << "\n";
              // for (int q=0;q<_q;q++){
              //   // problem was that was allowing some nan metadata as long as others provided - now just said not present in this case
              //   Rcpp::Rcout << "tau1[" << i << "][" << q << "] " << _tau1[i][q] <<"\n";
              //   Rcpp::Rcout << "taum[" << t << "][" << i << "][" << q << "] " << tauMarginal(t,i,q) <<"\n";
              // }
              
              LKL += log(_trans[groupstm1.at(i)][groupst.at(i)]);
            }
            catch (const std::out_of_range& oor) {
              Rcpp::Rcout << "t " << t << " i " << i << "\n";
              Rcpp::Rcout << "groupstm1[i] " << groupstm1.at(i) << "\n";
              Rcpp::Rcout << "groupst[i] " << groupst.at(i) << "\n";
              std::cerr << "Out of Range error: " << oor.what() << '\n';
            }
          }
          else{
            try{
              // Rcpp::Rcout << "t " << t << " i " << i << "\n";
              // Rcpp::Rcout << "groupst[i] " << groupst.at(i);
              LKL += log(_stationary[groupst.at(i)]);
            }
            catch (const std::out_of_range& oor) {
              Rcpp::Rcout << "t " << t << " i " << i << "\n";
              Rcpp::Rcout << "groupst[i] " << groupst.at(i) << "\n";
              std::cerr << "Out of Range error: " << oor.what() << '\n';
            }
          }
	      }
      groupstm1 = groupst;
    }
    //std::cerr<<LKL<<" ";
    // term 3
    for(int t=0;t<_t;t++){
      std::vector<int> groupst = getGroupsByMAP(t);
      for(int i=0;i<_n;i++)
        if (ispresent(t,i)){
          for(int j=0;j<i;j++)
            if (ispresent(t,j)){
              LKL += logDensity(t,groupst[i],groupst[j],Y[t][i][j]);
              if (_isdirected) LKL += logDensity(t,groupst[j],groupst[i],Y[t][j][i]);
            }	  
          if (_withselfloop){
            LKL += logDensity(t,groupst[i],groupst[i],Y[t][i][i]);
          }
        }
    }
    //std::cerr<<LKL<<std::endl;
    // term 4
    for(int t=0;t<_t;t++){
      std::vector<int> groupst = getGroupsByMAP(t);
      for(int i=0;i<_n;i++){
        if (ispresent(t,i)){ // necessary check as else groupst[i]=-1 and will throw error
          for(int s=0;s<_s;s++){
            if (ismetapresent(t,i,s)){
              LKL += logMetadensity(t,groupst[i],s,X[t][i][s]);
            }
          }
        }
      }
    }
    return(LKL);
  }
  
  template<typename Ytype, typename Xtype> 
  void DynSBM<Ytype,Xtype>::updateTauMarginal(){
    for(int t=1;t<_t;t++){
      int tstorage = t-1;
      for(int i=0;i<_n;i++){
        if (ispresent(t-1,i) && ispresent(t,i)){
          for(int qprime=0;qprime<_q;qprime++){
            _taum[tstorage][i][qprime] = 0.;
            for(int q=0;q<_q;q++){
              _taum[tstorage][i][qprime] += tauMarginal(t-1,i,q)*_taut[tstorage][i][q][qprime];
            }
          }
        }
      }
	  } 
    correctTaum(); // TODO correctTaum(t) would be better
  }
  
  
  template<typename Ytype, typename Xtype> 
  double DynSBM<Ytype, Xtype>::tauMarginal(int t, int i, int q) const{
    if(t==0){
      return(_tau1[i][q]);
    } else{
      if(ispresent(t-1,i)){
        int tstorage = t-1;
        return(_taum[tstorage][i][q]);
      } else {
	      return tauArrival(t,i,q);
      }
    }
  }
  
  template<typename Ytype, typename Xtype> 
  void DynSBM<Ytype, Xtype>::correctTau1(){// numerical issue : avoid too small value for tau
    for(int i=0;i<_n;i++){
      double sumtau1i = 0.;
      for(int q=0;q<_q;q++){
        if(_tau1[i][q]<precision){
          _tau1[i][q] = precision;
        }
        sumtau1i += _tau1[i][q];
      }
      for(int q=0;q<_q;q++){
	      _tau1[i][q] = _tau1[i][q]/sumtau1i;
      }
    }
  }
  
  template<typename Ytype, typename Xtype> 
  void DynSBM<Ytype,Xtype>::correctTaut(){// numerical issue : avoid too small value for tau
    for(int t=1;t<_t;t++){
      int tstorage = t-1;       
      for(int i=0;i<_n;i++){
        for(int q=0;q<(ispresent(t-1,i)?_q:1);q++){ // only q=0 if absent at t-1 (see tauArrival method)
          double sumtaut = 0.;
          for(int qprime=0;qprime<_q;qprime++){
            if(_taut[tstorage][i][q][qprime]<precision){
              _taut[tstorage][i][q][qprime] = precision;
            }
            sumtaut += _taut[tstorage][i][q][qprime];
          }
          for(int qprime=0;qprime<_q;qprime++){
            _taut[tstorage][i][q][qprime] = _taut[tstorage][i][q][qprime]/sumtaut;
          }
        }
      }
    }
  }
  
  template<typename Ytype, typename Xtype> 
  void DynSBM<Ytype,Xtype>::correctTaum(){// numerical issue : avoid too small value for tau
    for(int t=1;t<_t;t++){
      int tstorage = t-1;       
      for(int i=0;i<_n;i++){
        double sumtaum = 0.;
        for(int q=0;q<_q;q++){
          if(_taum[tstorage][i][q]<precision){
            _taum[tstorage][i][q] = precision;	    
          }
          sumtaum += _taum[tstorage][i][q];
        }
        for(int q=0;q<_q;q++){
          _taum[tstorage][i][q] = _taum[tstorage][i][q]/sumtaum;
        }
      }
    }
  }
  
  template<typename Ytype,typename Xtype> 
  void DynSBM<Ytype,Xtype>::initTau(const std::vector<int>& clustering){// initialization from a clustering
  // modify for metadata / use spectral algorithm?
    // it supposes previous allocation with O values
    for(int i=0;i<_n;i++){
      _tau1[i][clustering[i]] = 1;
    }
    correctTau1();
    for(int t=1;t<_t;t++){
      int tstorage = t-1;
      for(int i=0;i<_n;i++){
        for(int q=0;q<_q;q++){
          _taut[tstorage][i][q][clustering[i]] = 1;
        }
      }
    }
    correctTaut();
    updateTauMarginal();
  }
  
  template<typename Ytype, typename Xtype> 
  std::vector<int> DynSBM<Ytype,Xtype>::getGroupsByMAP(int t) const{ // maximum a posteriori - classification step
    std::vector<int> groups(_n,-1);
    if(t==0){
      for(int i=0;i<_n;i++){
        if(ispresent(0,i)){
          double maxi = 0.;
          for(int q=0;q<_q;q++){
            if(_tau1[i][q]>maxi){
              maxi = _tau1[i][q];
              groups[i] = q;
            }
          }
        }
      }
    } else{
      for(int i=0;i<_n;i++){
        if(ispresent(t,i)){
          double maxi = 0.;
          for(int q=0;q<_q;q++){
            double tau = tauMarginal(t,i,q);
            if(tau>maxi){
              maxi = tau;
              groups[i] = q;
            }
          }
        }
      }
    }
    return(groups);
  }
  
  template<typename Ytype, typename Xtype>
  void DynSBM<Ytype,Xtype>::correctBeta(){//numerical issue : avoid too small value for beta
    for(int t=0;t<_t;t++){
      for(int q=0;q<_q;q++){
        for(int l=0;l<_q;l++){
          if(_betaql[t][q][l]<precision){
            _betaql[t][q][l] = precision;
          }
          if(_betaql[t][q][l]>(1-precision)){
            _betaql[t][q][l] = 1-precision;
          }
          double beta = _betaql[t][q][l];
          _betaql[t][q][l] = log(beta); // trick: store log(betaql)
          _1minusbetaql[t][q][l] = log(1-beta); // trick: store log(1-betaql)
        }
      }
    }
  }

  template<typename Ytype, typename Xtype>
  double DynSBM<Ytype,Xtype>::logMetadensity(int t, int q, int s, Xtype x) const{
    std::string distname = _metatypes[s];
    std::vector<double> params = _varphi[t][q][s];
    return logMetalkl(distname,params,x);
  }



  template<typename Ytype, typename Xtype>
  void DynSBM<Ytype, Xtype>::updateVarphi(Xtype*** const X){
#pragma omp parallel for
    // need to rezero just in case, as for beta etc
    for(int t=0;t<_t;t++){
      for(int q=0;q<_q;q++){
        for(int s=0;s<_s;s++){
          for(int ds=0;ds<_distdims[s];ds++){
             _varphi[t][q][s][ds]=0.;
          }
        }
      }    
    }

    for (int s=0; s<_s; s++){
      // make sure have checked already by this point that distribution is one of accepted types
      // should also check if distnames contains whichever distribution requires certain parameters
      // i.e. don't need rho if only poisson or negative binomial, and if don't have either of them
      // then don't need zeta, and if don't have normal then don't need psi
      // for now know that will have at least poisson and categorical, so need 
      std::string distname = _metatypes[s];
      for(int t=0;t<_t;t++){
        std::vector<double> xit(_q,0.);
        // std::vector<double> zetat(_q,0.); // not needed - effectively the same as rhot 
        std::vector<std::vector<double>> rhot(_q,std::vector<double>(_metadims[s],0.));
        for(int i=0;i<_n;i++){  
          if (ispresent(t,i)){
            if (ismetapresent(t,i,s)){
              for(int q=0;q<_q;q++){
                xit[q]+=tauMarginal(t,i,q);
                for (int ds=0; ds<_metadims[s];ds++){
                  // zetat[q]+=_taum[t][i][q]*X[t][i][s][ds]; // not needed - effectively the same as rhot
                  rhot[q][ds]+=tauMarginal(t,i,q)*X[t][i][s][ds];
                } // ds
              } // q
            } // ismetapresent
          } // ispresent
        } // i
        // Rcpp::Rcout << "rho: \n";
        // for (auto & vec : rhot){
        //   for (auto & val: vec){
        //     Rcpp::Rcout << val << "\n";
        //   }
        // }
        // Rcpp::Rcout << "xi: \n";
        // for (auto & val: xit){
        //   Rcpp::Rcout << val << "\n";
        // }
        if (distname=="poisson"){
          // Rcpp::Rcout << "Updating poisson variational parameters...\n";
          for (int q=0;q<_q;q++){
            if (xit[q]!=0.){
              for (int ds=0;ds<_metadims[s];ds++){
                _varphi[t][q][s][ds]=rhot[q][ds]/xit[q];
              } // ds
            } 
          } // q
        }else if (distname=="negative binomial"){
          for (int q=0;q<_q;q++){
            for (int ds=0;ds<_metadims[s];ds++){
              throw std::invalid_argument("Still need to implement inference for neg. bin.");
              // _varphi[t][q][s][ds]= // r - need to implement - Newton's method?
              // _varphi[t][q][s][ds+_metadims[s]]= // p - need to implement - Newton's method?
            } // ds
          } // q
        }else if (str_contains(distname,"categorical")){
          for (int q=0;q<_q;q++){
            if (xit[q]!=0.){
              for (int ds=0;ds<_metadims[s];ds++){
                _varphi[t][q][s][ds]=rhot[q][ds]/xit[q];
              } // ds
            } 
          } // q
        }else if (str_contains(distname,"independent bernoulli")){
          // Rcpp::Rcout << "Updating independent Bernoulli variational parameters...\n";
          for (int q=0;q<_q;q++){
            if (xit[q]!=0.){
              for (int ds=0;ds<_metadims[s];ds++){  
                _varphi[t][q][s][ds]=rhot[q][ds]/xit[q];
              } // ds
            } 
          } // q
        }else if (str_contains(distname,"normal")){
          std::vector<std::vector<double>> psit(_q,std::vector<double>(_metadims[s],0.));
          for(int i=0;i<_n;i++){  
            if (ispresent(t,i)){
              if (ismetapresent(t,i,s)){
                for(int q=0;q<_q;q++){
                  xit[q]+=tauMarginal(t,i,q);
                  for (int ds=0; ds<_metadims[s];ds++){
                    psit[q][ds]+=tauMarginal(t,i,q)*X[t][i][s][ds]*X[t][i][s][ds];
                  }
                }
              }
            }
          }
          if (str_contains(distname,"independent")){
            for (int q=0;q<_q;q++){
              if (xit[q]!=0.){
                for (int ds=0;ds<_metadims[s];ds++){
                  _varphi[t][q][s][ds]=rhot[q][ds]/xit[q]; // mean
                  _varphi[t][q][s][ds+_metadims[s]]=(psit[q][ds]-2*_varphi[t][q][s][ds]*rhot[q][ds]+xit[q]*_varphi[t][q][s][ds]*_varphi[t][q][s][ds])/xit[q]; // variance
                } // ds
              }
            } // q
          }else if (str_contains(distname,"shared")){
            for (int q=0;q<_q;q++){
              _varphi[t][q][s].back() = 0.; // shared variance
              if (xit[q]!=0.){
                for (int ds=0;ds<_metadims[s];ds++){
                  _varphi[t][q][s][ds]=rhot[q][ds]/xit[q]; // mean
                  _varphi[t][q][s].back()+=(psit[q][ds]-2*_varphi[t][q][s][ds]*rhot[q][ds]+xit[q]*_varphi[t][q][s][ds]*_varphi[t][q][s][ds])/xit[q]; // variance
                }
              }
            }
          }else{
            throw std::invalid_argument("Multivariate normal not yet implemented");
          }
        }else{
          throw std::invalid_argument("Unrecognised distribution passed:\nChoose from poisson, negative binomial,\ncategorical x L, independent bernoulli x L, (shared/independent) normal");
        }
      } // t
    } // s
            
    correctVarphi();
  }

  template<typename Ytype, typename Xtype>
  void DynSBM<Ytype,Xtype>::correctVarphi(){//numerical issue : avoid too small value for varphi
    for(int t=0;t<_t;t++){
      for(int q=0;q<_q;q++){
        for(int s=0;s<_s;s++){
          
          // try{
          std::vector<double> tvarphi = _varphi[t][q][s];
          _varphi[t][q][s] = vecthresh(tvarphi,precision);
          // } catch (std::exception &e){
          //   std::cout << "Standard exception: " << e.what() << std::endl;
          // }
        }
      }
    }
  }
}
#endif
