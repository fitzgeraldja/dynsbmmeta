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
#include<DynSBMPoisson.h>
namespace dynsbm{
  void DynSBMPoisson::updateTheta(int*** const Y){// M-step
    for(int t=0;t<_t;t++)
      for(int q=0;q<_q;q++)
        for(int l=0;l<_q;l++){
            _ztplamql[t][q][l] = 1.;
            _ztplamqlnum[t][q][l] = 0.;
            _ztplamqlden[t][q][l] = 0.;
        }
    
    DynSBMPoissonAddEventFunctor addEventFunctor(*this);
    updateThetaCore<DynSBMPoissonAddEventFunctor>(Y, addEventFunctor); // this provide num and den along w beta
    
    for(int t=0;t<_t;t++){// invert to get lambdas
      for(int q=(_isdirected?0:1);q<_q;q++){
        for(int l=0;l<q;l++){
          if (_ztplamqlden[t][q][l]>0)
            if (_ztplamqlnum[t][q][l]>0){
              _ztplamql[t][q][l] = invPsi(_ztplamqlnum[t][q][l]/_ztplamqlden[t][q][l]);
              if(!_isdirected) _ztplamql[t][l][q] = _ztplamql[t][q][l];
            }
	      }        	
        if(_isdirected)
          for(int l=q+1;l<_q;l++){
            if (_ztplamqlden[t][q][l]>0)
              if (_ztplamqlnum[t][q][l]>0)
                _ztplamql[t][q][l] = invPsi(_ztplamqlnum[t][q][l]/_ztplamqlden[t][q][l]);
          }      
      }
    }
    for(int q=0;q<_q;q++){// invert to get lambdas
      double sumlamnumq = 0.;
      double sumlamdenq = 0.;
      double lamqres = 0.;
      for(int t=0;t<_t;t++){
        sumlamnumq += _ztplamqlnum[t][q][q];
        sumlamdenq += _ztplamqlden[t][q][q];
      }
      if (sumlamdenq>0){
        double lamqres = invPsi(sumlamnumq/sumlamdenq);  
        for(int t=0;t<_t;t++)
          if (lamqres>0){
            // Rcpp::Rcout << "Estimated lam for q: " << lamqres << "\n";
            _ztplamql[t][q][q] = lamqres;   
          }   
      }
    }
    correctZTPlam();
  }
  
  void DynSBMPoisson::updateFrozenTheta(int*** const Y){// M-step
    for(int t=0;t<_t;t++)
      for(int q=0;q<_q;q++)
        for(int l=0;l<_q;l++){
            _ztplamql[t][q][l] = 1.;
            _ztplamqlnum[t][q][l] = 0.;
            _ztplamqlden[t][q][l] = 0.;
        }
          
    
    DynSBMPoissonAddEventFunctor addEventFunctor(*this);
    updateFrozenThetaCore<DynSBMPoissonAddEventFunctor>(Y, addEventFunctor);
    
    for(int t=0;t<_t;t++){// invert to get lambdas
      for(int q=(_isdirected?0:1);q<_q;q++){
        for(int l=0;l<q;l++){
          if (_ztplamqlden[t][q][l]>0)
            if (_ztplamqlnum[t][q][l]>0){
              _ztplamql[t][q][l] = invPsi(_ztplamqlnum[t][q][l]/_ztplamqlden[t][q][l]);
              if(!_isdirected) _ztplamql[t][l][q] = _ztplamql[t][q][l];
            }
	      }        	
        if(_isdirected)
          for(int l=q+1;l<_q;l++){
            if (_ztplamqlden[t][q][l]>0)
              if (_ztplamqlnum[t][q][l]>0)
                _ztplamql[t][q][l] = invPsi(_ztplamqlnum[t][q][l]/_ztplamqlden[t][q][l]);
          }      
      }
    }
    for(int q=0;q<_q;q++){// invert to get lambdas
      double sumlamnumq = 0.;
      double sumlamdenq = 0.;
      double lamqres = 0.;
      for(int t=0;t<_t;t++){
        sumlamnumq += _ztplamqlnum[t][q][q];
        sumlamdenq += _ztplamqlden[t][q][q];
      }
      if (sumlamdenq>0){
        double lamqres = invPsi(sumlamnumq/sumlamdenq);  
        for(int t=0;t<_t;t++)
          if (lamqres>0)
            _ztplamql[t][q][q] = lamqres;    
      }  
    }
    correctZTPlam();
  }
  
  void DynSBMPoisson::correctZTPlam(){ // numerical issue : avoid too small value for ztplam
    for(int t=0;t<_t;t++){
        for(int q=0;q<_q;q++){
            for(int l=0;l<_q;l++){
                // don't think necessary to modify these at this point - if anything do before calling invPsi
                // if (_ztplamqlden[t][q][l]<precision){
                //     _ztplamqlden[t][q][l] = precision;
                // }
                // if (_ztplamqlnum[t][q][l]<precision){
                //     _ztplamqlnum[t][q][l] = precision;
                // }
                if (_ztplamql[t][q][l]<precision){
                    _ztplamql[t][q][l] = precision;
                }
		//   // trick: store log(_multinomprobaql[t][q][l][k])
		//    _multinomprobaql[t][q][l][k] = log(_multinomprobaql[t][q][l][k]);
                // }
            }
        }
    }
  }
}
