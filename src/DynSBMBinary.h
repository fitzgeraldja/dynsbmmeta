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
=======
  along with dynsbmmeta.  If not, see <http://www.gnu.org/licenses/>
>>>>>>> dynsbmmeta-cp/master
*/
#ifndef DYNSBM_DYNSBMBINARY_H
#define DYNSBM_DYNSBMBINARY_H
#include<DynSBM.h>
namespace dynsbm{
  class DynSBMBinary 
<<<<<<< HEAD
    : public DynSBM<int>{
  public:
    DynSBMBinary(int T, int N, int Q, const Rcpp::IntegerMatrix & present, bool isdirected = false, bool withselfloop = false)
      : DynSBM<int>(T,N,Q,present,isdirected,withselfloop) {}
    ~DynSBMBinary(){}
    virtual double logDensity(int t, int q, int l, int y) const{
      if(y==0){
	return(_betaql[t][q][l]); // trick: which is actually log(_betaql[t][q][l]))
      } else{
	return(_1minusbetaql[t][q][l]); // trick: which is actually log(1-_betaql[t][q][l]))
=======
    : public DynSBM<int,std::vector<double>>{
  public:
    DynSBMBinary(int T, int N, int Q, int S, const Rcpp::IntegerMatrix & present, const std::vector<std::vector<std::vector<int>>> & metapresent, const std::vector<std::string> & metatypes, const std::vector<int> & metadims, const std::vector<double> & metatuning, bool isdirected = false, bool withselfloop = false)
      : DynSBM<int,std::vector<double>>(T,N,Q,S,present,metapresent,metatypes,metadims,metatuning,isdirected,withselfloop) {}
    ~DynSBMBinary(){}
    virtual double logDensity(int t, int q, int l, int y) const{
      if(y==0){
	      return(_betaql[t][q][l]); // trick: which is actually log(_betaql[t][q][l]))
      } else{
	      return(_1minusbetaql[t][q][l]); // trick: which is actually log(1-_betaql[t][q][l]))
>>>>>>> dynsbmmeta-cp/master
      }
    }
    virtual void updateTheta(int*** const Y);
    virtual void updateFrozenTheta(int*** const Y);
  };
  
  class DynSBMBinaryAddEventFunctor{
    DynSBMBinary& _dynsbm;
  public:
    DynSBMBinaryAddEventFunctor(DynSBMBinary& dynsbm)
      : _dynsbm(dynsbm) {}
    void operator()(double proba, int y, int t, int q, int l){}
  };
}
#endif
