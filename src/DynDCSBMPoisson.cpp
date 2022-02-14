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
  void DynDCSBMPoisson::updateThetaCore(int*** const Y){
    for(int t=0;t<_t;t++){
      for(int i=0;i<_n;i++){
        if (ispresent(t,i)){    
          for(int q=0;q<_q;q++){
            _dckappaq[t][q][0] += _degs[t][i][0]*tauMarginal(t,i,q); 
            _dckappaq[t][q][1] += _degs[t][i][1]*tauMarginal(t,i,q); 
            for(int j=0;j<i;j++){ // note j<i
              if (ispresent(t,j)){
                for(int l=0;l<q;l++){
                  if(_isdirected){
                    if (!std::isnan(Y[t][i][j])){		      
                      _dcmql[t][q][l] += tauMarginal(t,i,q)*tauMarginal(t,j,l)*Y[t][i][j];
                      _dcmql[t][l][q] += tauMarginal(t,i,l)*tauMarginal(t,j,q)*Y[t][i][j];
                    }
                    if (!std::isnan(Y[t][j][i])){
                      _dcmql[t][q][l] += tauMarginal(t,i,l)*tauMarginal(t,j,q)*Y[t][j][i];
                      _dcmql[t][l][q] += tauMarginal(t,i,q)*tauMarginal(t,j,l)*Y[t][j][i];
                    }
                  } else{
                    if (!std::isnan(Y[t][i][j])){
                      _dcmql[t][q][l] += tauMarginal(t,i,q)*tauMarginal(t,j,l)*Y[t][i][j];
                    }
                  }
                }
                // q==l
                if (!std::isnan(Y[t][i][j])){
                  _dcmql[t][q][q] += tauMarginal(t,i,q)*tauMarginal(t,j,q)*Y[t][i][j];
                }
                if(_isdirected){
                  if (!std::isnan(Y[t][j][i])){
                    _dcmql[t][q][q] += tauMarginal(t,i,q)*tauMarginal(t,j,q)*Y[t][j][i];
                  }
                }
              }
            }
            // j==i considered only if selfloop allowed
            if (_withselfloop){
              _dcmql[t][q][q] += tauMarginal(t,i,q)*tauMarginal(t,i,q)*Y[t][i][i];
            }
          }
        }	
      }
    }
    for(int t=0;t<_t;t++){// symmetrise if need be
      for(int q=0;q<_q;q++){
        for(int l=0;l<q;l++){
          if(!_isdirected) _dcmql[t][l][q] = _dcmql[t][q][l];        	
        }
      }
    }
  }

  void DynDCSBMPoisson::updateTheta(int*** const Y){// M-step
    for(int t=0;t<_t;t++){
      for(int q=0;q<_q;q++){
        _dckappaq[t][q][0] = 0.;
        _dckappaq[t][q][1] = 0.;
        _dclamql[t][q][q] = 0.;
        _dcmql[t][q][q] = 0.;
        for(int l=0;l<_q;l++){
          _dclamql[t][q][l] = 1.;
          _dcmql[t][q][l] = 0.;
          if (_isdirected){
            _dclamql[t][l][q] = 1.;
            _dcmql[t][l][q] = 0.;
          }  
        }
      }
    }
    updateThetaCore(Y); // this will provide our kappas and ms - note no beta now
    // update eqn in directed case is lam_ql = m_ql/kappa^out_q * kappa^in_l, and just 
    // plain kappas in undirected case
    
    // note the constraint lamql[t,q,q] = lamql[,q,q]
    for(int t=0;t<_t;t++){
      for(int q=(_isdirected?0:1);q<_q;q++){        
        for(int l=0;l<_q;l++){ // note l < q
          if (_isdirected){
            _dclamql[t][q][l] = _dcmql[t][q][l] / (_dckappaq[t][q][0] * _dckappaq[t][l][1]);
            _dclamql[t][l][q] = _dcmql[t][l][q] / (_dckappaq[t][l][0] * _dckappaq[t][q][1]);
          } else {
            _dclamql[t][q][l] = _dcmql[t][q][l] / (_dckappaq[t][q][0] * _dckappaq[t][l][0]);
          }
        }
      }
    }
    for(int q=0;q<_q;q++){
      double summq = 0.;
      double sumkappaqout = 0.;
      double sumkappaqin = 0.;
      for(int t=0;t<_t;t++){
        summq += _dcmql[t][q][q];
        sumkappaqout += _dckappaq[t][q][0];
        if (_isdirected) sumkappaqin += _dckappaq[t][q][1];
      }
      for(int t=0;t<_t;t++){
        if (_isdirected){
          _dclamql[t][q][q] = summq / (sumkappaqout * sumkappaqin);
        } else {
          _dclamql[t][q][q] = summq / (sumkappaqout * sumkappaqout);
        }
      }
    }
    correctDClam();
  }
  
  void DynDCSBMPoisson::updateFrozenTheta(int*** const Y){// M-step
    for(int t=0;t<_t;t++){
      for(int q=0;q<_q;q++){
        _dckappaq[t][q][0] = 0.;
        _dckappaq[t][q][1] = 0.;
        _dclamql[t][q][q] = 0.;
        _dcmql[t][q][q] = 0.;
        for(int l=0;l<_q;l++){
          _dclamql[t][q][l] = 1.;
          _dcmql[t][q][l] = 0.;
          if (_isdirected){
            _dclamql[t][l][q] = 1.;
            _dcmql[t][l][q] = 0.;
          }  
        }
      }
    }
    updateThetaCore(Y); // this will provide our kappas and ms - note no beta now
    // update eqn in directed case is lam_ql = m_ql/kappa^out_q * kappa^in_l, and just 
    // plain kappas in undirected case
    
    // note the constraint lamql[t,q,q] = lamql[,q,q]
    for(int t=0;t<_t;t++){
      for(int q=(_isdirected?0:1);q<_q;q++){        
        for(int l=0;l<_q;l++){ // note l < q
          if (_isdirected){
            _dclamql[t][q][l] = _dcmql[t][q][l] / (_dckappaq[t][q][0] * _dckappaq[t][l][1]);
            _dclamql[t][l][q] = _dcmql[t][l][q] / (_dckappaq[t][l][0] * _dckappaq[t][q][1]);
          } else {
            _dclamql[t][q][l] = _dcmql[t][q][l] / (_dckappaq[t][q][0] * _dckappaq[t][l][0]);
          }
        }
      }
    }
    for(int q=0;q<_q;q++){
      double summq = 0.;
      double sumkappaqout = 0.;
      double sumkappaqin = 0.;
      for(int t=0;t<_t;t++){
        summq += _dcmql[t][q][q];
        sumkappaqout += _dckappaq[t][q][0];
        if (_isdirected) sumkappaqin += _dckappaq[t][q][1];
      }
      for(int t=0;t<_t;t++){
        if (_isdirected){
          _dclamql[t][q][q] = summq / (sumkappaqout * sumkappaqin);
        } else {
          _dclamql[t][q][q] = summq / (sumkappaqout * sumkappaqout);
        }
      }
    }
    correctDClam();
  }

  double DynDCSBMPoisson::completedLoglikelihood(int*** const Y, std::vector<double>*** const X) const{ // including entropy term
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
                  if (_isdirected){
                    J += taumtiq*tauMarginal(t,j,l)*logDCDensity(t,q,l,Y[t][i][j],_degs[t][i][0],_degs[t][j][1]);
                    J += taumtiq*tauMarginal(t,j,l)*logDCDensity(t,l,q,Y[t][j][i],_degs[t][j][0],_degs[t][i][1]);
                  } else {
                    J += taumtiq*tauMarginal(t,j,l)*logDCDensity(t,q,l,Y[t][i][j],_degs[t][i][0],_degs[t][j][0]);
                  }
		            }
	            }
          if (_withselfloop){
            for(int q=0;q<_q;q++){
              if (_isdirected){
                J += tauMarginal(t,i,q)*logDCDensity(t,q,q,Y[t][i][i],_degs[t][i][0],_degs[t][i][1]);
              }else{
                J += tauMarginal(t,i,q)*logDCDensity(t,q,q,Y[t][i][i],_degs[t][i][0],_degs[t][i][0]);
              }
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

  double DynDCSBMPoisson::modelselectionLoglikelihood(int*** const Y, std::vector<double>*** const X) const{
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
              if (_isdirected){
                LKL += logDCDensity(t,groupst[i],groupst[j],Y[t][i][j],_degs[t][i][0],_degs[t][j][1]);
                LKL += logDCDensity(t,groupst[j],groupst[i],Y[t][j][i],_degs[t][j][0],_degs[t][i][1]);
              } else {
                LKL += logDCDensity(t,groupst[i],groupst[j],Y[t][i][j],_degs[t][i][0],_degs[t][j][0]);
              }
            }	  
          if (_withselfloop){
            if (_isdirected){
              LKL += logDCDensity(t,groupst[i],groupst[i],Y[t][i][i],_degs[t][i][0],_degs[t][i][1]);
            } else {
              LKL += logDCDensity(t,groupst[i],groupst[i],Y[t][i][i],_degs[t][i][0],_degs[t][i][0]);
            }
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

  void DynDCSBMPoisson::updateTau(int*** const Y, std::vector<double>*** const X){
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
                if (_isdirected){
                  if (!std::isnan(Y[0][i][j])){
                    logp += _tau1[j][l]*logDCDensity(0,q,l,Y[0][i][j],_degs[0][i][0],_degs[0][j][1]);
                  }
                  if (!std::isnan(Y[0][j][i])){
                    logp += _tau1[j][l]*logDCDensity(0,l,q,Y[0][j][i],_degs[0][j][0],_degs[0][i][1]);
                  }
                } else {
                  if (!std::isnan(Y[0][i][j])){
                    logp += _tau1[j][l]*logDCDensity(0,q,l,Y[0][i][j],_degs[0][i][0],_degs[0][j][0]);
                  }
                }
              }
            }
          }
          if((_withselfloop)&(!std::isnan(Y[0][i][i]))){
            if (_isdirected)
              logp += logDCDensity(0,q,q,Y[0][i][i],_degs[0][i][0],_degs[0][i][1]); 
            else 
              logp += logDCDensity(0,q,q,Y[0][i][i],_degs[0][i][0],_degs[0][i][0]); 
          } 
          for(int j=i+1;j<_n;j++){
            if (ispresent(0,j)){
              for(int l=0;l<_q;l++){
                if (_isdirected){
                  if (!std::isnan(Y[0][i][j])){
                    logp += _tau1[j][l]*logDCDensity(0,q,l,Y[0][i][j],_degs[0][i][0],_degs[0][j][1]);
                  }
                  if (!std::isnan(Y[0][j][i])){
                    logp += _tau1[j][l]*logDCDensity(0,l,q,Y[0][j][i],_degs[0][j][0],_degs[0][i][1]);
                  }
                } else {
                  if (!std::isnan(Y[0][i][j])){
                    logp += _tau1[j][l]*logDCDensity(0,q,l,Y[0][i][j],_degs[0][i][0],_degs[0][j][0]);
                  }
                }
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
                  if (_isdirected){
                    if (!std::isnan(Y[t][i][j])){
                      logp += tauMarginal(t,j,l)*logDCDensity(t,qprime,l,Y[t][i][j],_degs[t][i][0],_degs[t][j][1]);
                    }
                    if (!std::isnan(Y[t][j][i])){
                      logp += tauMarginal(t,j,l)*logDCDensity(t,l,qprime,Y[t][j][i],_degs[t][j][0],_degs[t][i][1]);
                    }
                  } else {
                    if (!std::isnan(Y[t][i][j])){
                      logp += tauMarginal(t,j,l)*logDCDensity(t,qprime,l,Y[t][i][j],_degs[t][i][0],_degs[t][j][0]);
                    }
                  }
                }
              }
            }
	          if((_withselfloop)&(!std::isnan(Y[t][i][i]))) 
              if (_isdirected) 
                logp += logDCDensity(t,qprime,qprime,Y[t][i][i],_degs[t][i][0],_degs[t][i][1]);
              else 
                logp += logDCDensity(t,qprime,qprime,Y[t][i][i],_degs[t][i][0],_degs[t][i][0]);
            for(int j=i+1;j<_n;j++){
              if (ispresent(t,j)){
                for(int l=0;l<_q;l++){
                  if (_isdirected){
                    if (!std::isnan(Y[t][i][j])){
                      logp += tauMarginal(t,j,l)*logDCDensity(t,qprime,l,Y[t][i][j],_degs[t][i][0],_degs[t][j][1]);
                    }
                    if (!std::isnan(Y[t][j][i])){
                      logp += tauMarginal(t,j,l)*logDCDensity(t,l,qprime,Y[t][j][i],_degs[t][j][0],_degs[t][i][1]);
                    }
                  } else {
                    if (!std::isnan(Y[t][i][j])){
                      logp += tauMarginal(t,j,l)*logDCDensity(t,qprime,l,Y[t][i][j],_degs[t][i][0],_degs[t][j][0]);
                    }
                  }
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
            if (!std::isnan(Y[t][i][j])) _degs[t][i][0] += Y[t][i][j];
            if (_isdirected){
              if (!std::isnan(Y[t][j][i])) _degs[t][i][1] += Y[t][j][i];
            }
          }
          if (_isdirected){
            for(int j=i+1;j<_n;j++){
              if (!std::isnan(Y[t][i][j])) _degs[t][i][0] += Y[t][i][j];
              if (!std::isnan(Y[t][j][i])) _degs[t][i][1] += Y[t][j][i];
            }
          }
          if (_withselfloop){
            if (!std::isnan(Y[t][i][i])) _degs[t][i][0] += Y[t][i][i];
          }
        }
      }
    }
  }
}
