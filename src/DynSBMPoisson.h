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
#ifndef DYNSBM_DYNSBMPOISSON_H
#define DYNSBM_DYNSBMPOISSON_H
#include<DynSBM.h>
//#include<boost/math/distributions/poisson.hpp>
#include<boost/math/special_functions/lambert_w.hpp>

namespace dynsbm{
  using boost::math::poisson;
  using boost::math::lambert_w0;
  class DynSBMPoisson 
    : public DynSBM<int,std::vector<double>>{
  protected:
    
    double*** _ztplamql; // poisson mean for edges between q and l
    double*** _ztplamqlnum; // numerator inside inverse function for lambda
    double*** _ztplamqlden; // denominator ...
    // void correctZTPinv();
    void correctZTPlam();
    
    poisson*** _ztpdists;
    // void addEvent(double proba, int y, int t, int q){
    //   // fix
    //   _multinomprobaql[t][q][l][y-1] += proba; // y is an integer value in [1,2,...K]
    // }
  public:
    DynSBMPoisson(int T, int N, int Q, int S, const Rcpp::IntegerMatrix & present, const std::vector<std::vector<std::vector<int>>> & metapresent, const std::vector<std::string> & metatypes, const std::vector<int> & metadims, const std::vector<double> & metatuning, bool isdirected = false, bool withselfloop = false)
      : DynSBM<int,std::vector<double>>(T,N,Q,S,present,metapresent,metatypes,metadims,metatuning,isdirected,withselfloop) {
      allocate3D(_ztplamql,_t,_q,_q);
      // need to set nonzero, choose as 1 arbitrarily -- will immediately be corrected on first iteration
      for (int t=0;t<_t;t++){
        for (int q=0;q<_q;q++){
          for (int l=0;l<_q;l++){
            _ztplamql[t][q][l] = 1.;
          }
        }
      }
      allocate3D(_ztplamqlnum,_t,_q,_q);
      allocate3D(_ztplamqlden,_t,_q,_q);
      allocate3D(_ztpdists,_t,_q,_q,1.0);
      // Rcpp::Rcout << "Successfully allocated distributions" << "\n";
    }
    ~DynSBMPoisson(){
      deallocate3D(_ztplamql,_t,_q,_q);
      deallocate3D(_ztplamqlnum,_t,_q,_q);
      deallocate3D(_ztplamqlden,_t,_q,_q);
      deallocate3D(_ztpdists,_t,_q,_q);
    }
    
    double invPsi(double val){
      // take num/den as val
      return(lambert_w0(-1.0*exp(-1.0*val)*val)+val);
    }

    double getZTPlam(int t, int q, int l) const{
      return(_ztplamql[t][q][l]);
    }
    
    void makeZTPdists(){
        for (int t=0;t<_t;t++){
            for (int q=0;q<_q;q++){
                for (int l=0;l<_q;l++){
                    _ztpdists[t][q][l]=poisson(_ztplamql[t][q][l]);
                }
            }
        }
    }
    
    virtual double logDensity(int t, int q, int l, int y) const{
      
      if(y==0){
	      return(_betaql[t][q][l]); // trick: which is actually log(_betaql[t][q][l]))
      } else{
        // NB: y is an integer value
        return(_1minusbetaql[t][q][l] // trick: which is actually log(1-_betaql[t][q][l]))
	       //+ log(_multinomprobaql[t][q][l][y-1]));
	       + log(pdf(_ztpdists[t][q][l],y))); 
      }
    }
    virtual void updateTheta(int*** const Y);
    virtual void updateFrozenTheta(int*** const Y);
    friend class DynSBMPoissonAddEventFunctor;
  };

  class DynSBMPoissonAddEventFunctor{
    DynSBMPoisson& _dynsbmmeta;
    public:
      DynSBMPoissonAddEventFunctor(DynSBMPoisson& dynsbmmeta)
        : _dynsbmmeta(dynsbmmeta) {}
      void operator()(double proba, int y, int t, int q, int l){
        _dynsbmmeta._ztplamqlnum[t][q][l] += proba*y; // y is an integer value
        _dynsbmmeta._ztplamqlden[t][q][l] += proba;
      }
  };
}
#endif
