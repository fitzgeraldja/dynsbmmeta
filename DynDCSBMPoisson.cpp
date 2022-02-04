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
#include<DynDCSBMPoisson.h>
namespace dynsbm{
  void DynDCSBMPoisson::updateTheta(int*** const Y, int** const D){// M-step
    for(int t=0;t<_t;t++)
      for(int q=0;q<_q;q++)
        for(int l=0;l<_q;l++){
            _dclamql[t][q][l] = 1.;
            _dcmql[t][q][l] = 0.;
            _dckappaq[t][q] = 0.;
        }
    
    // DynDCSBMPoissonAddEventFunctor addEventFunctor(*this);
    // updateThetaCore<DynDCSBMPoissonAddEventFunctor>(Y, addEventFunctor); // this provide num and den along w beta
    
    for(int t=0;t<_t;t++){
      for(int q=(_isdirected?0:1);q<_q;q++){
        for(int l=0;l<q;l++){
          _dclamql[t][q][l] = _dcmql[t][q][l]/(_dckappaq[t][q]*_dckappaq[t][l]);
          if(!_isdirected) _dclamql[t][l][q] = _dclamql[t][q][l];
	      }        	
        if(_isdirected)
          for(int l=q+1;l<_q;l++){
            _dclamql[t][q][l] = _dcmql[t][q][l]/(_dckappaq[t][q]*_dckappaq[t][l]);
          }      
      }
    }
    for(int q=0;q<_q;q++){
      double sumlamnumq = 0.;
      double sumlamdenq = 0.;
      double lamqres = 0.;
      for(int t=0;t<_t;t++){
        sumlamnumq += _dclamqlnum[t][q][q];
        sumlamdenq += _dclamqlden[t][q][q];
      }
      if (sumlamdenq>0){
        double lamqres = invPsi(sumlamnumq/sumlamdenq);  
        for(int t=0;t<_t;t++)
          if (lamqres>0){
            // Rcpp::Rcout << "Estimated lam for q: " << lamqres << "\n";
            _dclamql[t][q][q] = lamqres;   
          }   
      }
    }
    correctDClam();
  }
  
  void DynDCSBMPoisson::updateFrozenTheta(int*** const Y){// M-step
    for(int t=0;t<_t;t++)
      for(int q=0;q<_q;q++)
        for(int l=0;l<_q;l++){
            _dclamql[t][q][l] = 1.;
            _dclamqlnum[t][q][l] = 0.;
            _dclamqlden[t][q][l] = 0.;
        }
          
    
    DynDCSBMPoissonAddEventFunctor addEventFunctor(*this);
    updateFrozenThetaCore<DynDCSBMPoissonAddEventFunctor>(Y, addEventFunctor);
    
    for(int t=0;t<_t;t++){// invert to get lambdas
      for(int q=(_isdirected?0:1);q<_q;q++){
        for(int l=0;l<q;l++){
          if (_dclamqlden[t][q][l]>0)
            if (_dclamqlnum[t][q][l]>0){
              _dclamql[t][q][l] = invPsi(_dclamqlnum[t][q][l]/_dclamqlden[t][q][l]);
              if(!_isdirected) _dclamql[t][l][q] = _dclamql[t][q][l];
            }
	      }        	
        if(_isdirected)
          for(int l=q+1;l<_q;l++){
            if (_dclamqlden[t][q][l]>0)
              if (_dclamqlnum[t][q][l]>0)
                _dclamql[t][q][l] = invPsi(_dclamqlnum[t][q][l]/_dclamqlden[t][q][l]);
          }      
      }
    }
    for(int q=0;q<_q;q++){// invert to get lambdas
      double sumlamnumq = 0.;
      double sumlamdenq = 0.;
      double lamqres = 0.;
      for(int t=0;t<_t;t++){
        sumlamnumq += _dclamqlnum[t][q][q];
        sumlamdenq += _dclamqlden[t][q][q];
      }
      if (sumlamdenq>0){
        double lamqres = invPsi(sumlamnumq/sumlamdenq);  
        for(int t=0;t<_t;t++)
          if (lamqres>0)
            _dclamql[t][q][q] = lamqres;    
      }  
    }
    correctDClam();
  }
  
  void DynDCSBMPoisson::correctDClam(){ // numerical issue : avoid too small value for dclam
    for(int t=0;t<_t;t++){
        for(int q=0;q<_q;q++){
            for(int l=0;l<_q;l++){
                // don't think necessary to modify these at this point - if anything do before calling invPsi
                // if (_dclamqlden[t][q][l]<precision){
                //     _dclamqlden[t][q][l] = precision;
                // }
                // if (_dclamqlnum[t][q][l]<precision){
                //     _dclamqlnum[t][q][l] = precision;
                // }
                if (_dclamql[t][q][l]<precision){
                    _dclamql[t][q][l] = precision;
                }
		//   // trick: store log(_multinomprobaql[t][q][l][k])
		//    _multinomprobaql[t][q][l][k] = log(_multinomprobaql[t][q][l][k]);
                // }
            }
        }
    }
  }

  void DynDCSBMPoisson::initDegs(int*** const Y){
    for(int t=0;t<_t;t++){
      for(int i=0;i<_n;i++){
        if(ispresent(t,i)){
          for(int j=0;j<i;j++){
            _degs[t][i][0] +=_Y[t][i][j];
            if (_isdirected){
              _degs[t][i][1] += Y[t][j][i];
            }
          }
          if (_withselfloop){
            _degs[t][i][0] += 1.0;
          }
        }
      }
    }
  }
}
