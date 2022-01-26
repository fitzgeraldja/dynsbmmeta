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
#ifndef DYNSBM_DynDCSBMPoisson_H
#define DYNSBM_DynDCSBMPoisson_H
#include<DynSBM.h>
//#include<boost/math/distributions/poisson.hpp>
// #include<boost/math/special_functions/lambert_w.hpp>

namespace dynsbm{
  using boost::math::poisson;
  // using boost::math::lambert_w0;
  class DynDCSBMPoisson 
    : public DynSBM<int,std::vector<double>>{
  protected:
    
    double*** _dclamql; // param for edges between q and l
    double*** _dcmql; // weighted edge count between groups 
    double** _dckappaq; // weighted total group degree
    // void correctDCinv();
    void correctDClam();
    
    // poisson*** _dcdists;
    // void addEvent(double proba, int y, int t, int q){
    //   // fix
    //   _multinomprobaql[t][q][l][y-1] += proba; // y is an integer value in [1,2,...K]
    // }
  public:
    DynDCSBMPoisson(int T, int N, int Q, int S, const Rcpp::IntegerMatrix & present, const std::vector<std::vector<std::vector<int>>> & metapresent, const std::vector<std::string> & metatypes, const std::vector<int> & metadims, const std::vector<double> & metatuning, bool isdirected = false, bool withselfloop = false)
      : DynSBM<int,std::vector<double>>(T,N,Q,S,present,metapresent,metatypes,metadims,metatuning,isdirected,withselfloop) {
      allocate3D(_dclamql,_t,_q,_q);
      // need to set nonzero, choose as 1 arbitrarily -- will immediately be corrected on first iteration
      for (int t=0;t<_t;t++){
        for (int q=0;q<_q;q++){
          for (int l=0;l<_q;l++){
            _dclamql[t][q][l] = 1.;
          }
        }
      }
      allocate3D(_dcmql,_t,_q,_q);
      allocate2D(_dckappaq,_t,_q);
      // allocate3D(_dcdists,_t,_q,_q,1.0);
      // Rcpp::Rcout << "Successfully allocated distributions" << "\n";
    }
    ~DynDCSBMPoisson(){
      deallocate3D(_dclamql,_t,_q,_q);
      deallocate3D(_dcmql,_t,_q,_q);
      deallocate2D(_dckappaq,_t,_q);
      // deallocate3D(_dcdists,_t,_q,_q);
    }
    
    // double invPsi(double val){
    //   // take num/den as val
    //   return(lambert_w0(-1.0*exp(-1.0*val)*val)+val);
    // }

    double getDClam(int t, int q, int l) const{
      return(_dclamql[t][q][l]);
    }
    
    // void makeDCdists(){
    //     for (int t=0;t<_t;t++){
    //         for (int q=0;q<_q;q++){
    //             for (int l=0;l<_q;l++){
    //                 _dcdists[t][q][l]=poisson(_dclamql[t][q][l]);
    //             }
    //         }
    //     }
    // }
    
    double logDCDensity(int t, int q, int l, int y, int di, int dj) const{
      return log(pdf(poisson(_dclamql[t][q][l]*di*dj),y));
    }
    void updateTheta(int*** const Y);
    void updateFrozenTheta(int*** const Y);
    // friend class DynDCSBMPoissonAddEventFunctor;
  };

  // class DynDCSBMPoissonAddEventFunctor{
  //   DynDCSBMPoisson& _dynsbm;
  //   public:
  //     DynDCSBMPoissonAddEventFunctor(DynDCSBMPoisson& dynsbm)
  //       : _dynsbm(dynsbm) {}
  //     void operator()(double proba, int y, int t, int q, int l){
  //       _dynsbm._dclamqlnum[t][q][l] += proba*y; // y is an integer value
  //       _dynsbm._dclamqlden[t][q][l] += proba;
  //     }
  // };
}
#endif
