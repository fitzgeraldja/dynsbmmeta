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
#ifndef DYNSBM_tOOLS
#define DYNSBM_tOOLS
#include <math.h>
#include <vector>
#include <algorithm> // std::set_intersection
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/normal.hpp>
#include <stdexcept>

// #include <Rcpp.h>
// Rcpp::NumericVector Rcpp::dnorm(Rcpp::NumericVector x, double mean, double sd, bool log)

namespace dynsbm{
  const double precision = 1e-10;
}

// need new function to find exponential of unknown sized vector
template<typename T>
auto vecexp(std::vector<T> xs) -> std::vector<T> {
  for(auto & x : xs){ x = std::exp(x); }
  return xs;
}
// need new function to find logarithm of unknown sized vector
template<typename T>
auto veclog(std::vector<T> xs) -> std::vector<T> {
  for(auto & x : xs){ x = std::log(x); }
  return xs;
}
// new function to threshold vectors
template<typename T>
auto vecthresh(std::vector<T> xs, double prcs) -> std::vector<T> {
  for(auto & x : xs){if ( x < prcs ){ x = prcs; }}
  return xs;
}


static bool str_contains(const std::string & str, const std::string & substr){
  if (str.find(substr)!=std::string::npos){
    return true;
  } else{
    return false;
  }
}
// bool str_contains(const Rcpp::String & str, const std::string & substr){
//   std::string str = Rcpp::as<std::string>(str);
//   if (str.find(substr)!=std::string::npos){
//     return true;
//   } else{
//     return false;
//   }
// }


// need new function that returns the probability of a particular piece of metadata x (as std::vector<double>) of type s at time t for group q
// can just pass a string distribution name, the distribution parameters (std::vector<double>) and the piece of metadata itself
static double indepbernlkl(const std::vector<double> & params, const std::vector<double>& x) {
  // x just one hot vector length L (?)
  // params are just L length vector with independent probability of observing that particular category
  // so just take product of params where x=1 (if one-hot) or at indices of x (if vector of codes)
  int L = params.size();
  double lkl = 1.;
  if (x.size()==L){
    // one-hot encoding
    for (int l=0; l<L; l++){
      lkl*=pow(params[l],x[l]);
      lkl*=pow(1-params[l],1-x[l]);
    }
  }
  else{
    // only passing codes
    std::vector<int> range_v;
    for(int i=0;i<L;i++){
      range_v.push_back(i);
    }
    // std::vector<int> inter_v(L);
    // std::vector<int>::iterator it;
    // it=std::set_intersection(x.begin(),x.end(),range_v.begin(),range_v.end(),inter_v);
    // inter_v.resize(it-inter_v.begin());
    std::vector<int> diff_v;
    std::set_difference(range_v.begin(),range_v.end(),x.begin(),x.end(),std::inserter(diff_v,diff_v.begin()));
    for (const auto & i: x){
      lkl*=params[i];
    }
    for (const auto & i: diff_v){
      lkl*=1-params[i];
    }
  }
  return lkl;
}
static double catlkl(const std::vector<double> & params, const std::vector<double>& x) {
  // x just int between 0 and l-1
  // params are just L length vector with probability of observing that particular category
  return params[x[0]];
}
static double poissonlkl(const std::vector<double> & params, const std::vector<double> & x) {
  using boost::math::poisson;
  double mean = params[0];
  if (!isnan(mean)){
    poisson poissondist(mean);
  return pdf(poissondist,x[0]);
  }else{
    return 0.;
  }
}
static double negbinlkl(const std::vector<double> & params, const std::vector<double> & x) {
  using boost::math::negative_binomial;
  double r = params[0];
  double p = params[1];
  negative_binomial nbdist(r,p);
  return pdf(nbdist,x[0]);
}
static double normallkl(const std::vector<double> & params, const std::vector<double> & x) {
  using boost::math::normal;
  int Ds = x.size();
  double lkl=1.;
  if (params.size()==Ds+1){
    // shared covariance case
    double sd = params.back();
    for (int i=0;i<Ds;i++){
      normal normaldist(params[i],sd);
      lkl*=pdf(normaldist,x[i]);
    }
  }else if (params.size()==2*Ds){
    // independent normal case
    // std::vector<double> sd(params.begin+Ds,params.end());
    for (int i=0;i<Ds;i++){
      normal normaldist(params[i],params[i+Ds]);
      lkl*=pdf(normaldist,x[i]);
    }
  }
  else{
    // multivariate normal case -- not implemented
    // see https://gallery.rcpp.org/articles/dmvnorm_arma/ for later implementation of multivariate case
  }
  return lkl;
}


static double logMetalkl(const std::string & distname, const std::vector<double> & params, const std::vector<double> & x) {
  if (distname=="poisson"){
    return log(poissonlkl(params,x));

  }else if (distname=="negative binomial"){
    return log(negbinlkl(params,x));

  }else if (str_contains(distname,"independent bernoulli")){
    return log(indepbernlkl(params,x));

  }else if (str_contains(distname,"categorical")){
    return log(catlkl(params,x));

  }else if (str_contains(distname,"normal")){
    return log(normallkl(params,x)); 

  }else{
    throw std::invalid_argument("Unrecognised distribution passed:\nChoose from poisson, negative binomial,\ncategorical x L, independent bernoulli x L, (shared/independent) normal");
  }
}

template<typename Ttype>
void allocate2D(Ttype**& ptr, int d1, int d2){
  ptr = new Ttype*[d1];
  ptr[0] = new Ttype[d1*d2];
  for (int p=0;p<d1*d2;p++) ptr[0][p] = Ttype(0);
  for (int i=1;i<d1;i++){
    ptr[i] = ptr[i-1] + d2;
  }
}
template<typename Ttype>
void deallocate2D(Ttype**& ptr, int d1, int d2){
  delete[] ptr[0];
  delete[] ptr;
}
template<typename Ttype>
void allocate3D(Ttype***& ptr, int d1, int d2, int d3, int init_val=0){
  ptr = new Ttype**[d1];
  for (int i=0;i<d1;i++){
    ptr[i] = new Ttype*[d2];
    for (int j=0;j<d2;j++){
      ptr[i][j] = new Ttype[d3];
      for(int k=0;k<d3;k++){
	      ptr[i][j][k] = Ttype(init_val);
      }
    }
  }
}
template<typename Ttype>
void allocate3Dvectors(std::vector<Ttype>***& ptr, int d1, int d2, int d3, const std::vector<int>& vecdims){
  ptr = new std::vector<Ttype>**[d1];
  for (int i=0;i<d1;i++){
    ptr[i] = new std::vector<Ttype>*[d2];
    for (int j=0;j<d2;j++){
      ptr[i][j] = new std::vector<Ttype>[d3];
      for(int k=0;k<d3;k++){
        ptr[i][j][k] = std::vector<Ttype>(vecdims[k],0);
      }
    }
  }
}
  /*
  for (int i=0;i<d1;i++){
    ptr[i] = new Ttype*[d2];
  }
  ptr[0][0] = new Ttype[d1*d2*d3];
  for (int p=0;p<d1*d2*d3;p++) ptr[0][0][p] = Ttype(0);
  int i=0, j=1;
  while(i<d1){
    if(j>0)
      ptr[i][j] = ptr[i][j-1]+d2;
    else
      ptr[i][j] = ptr[i-1][d2-1]+d2;
    j++;
    if(j==d2){
      j=0;i++;
    }
  }
  */

template<typename Ttype>
void deallocate3D(Ttype***& ptr, int d1, int d2, int d3){
  /*
    delete[] ptr[0][0];
    for (int i=0;i<d1;i++)
    delete[] ptr[i];
  */
  for (int i=0;i<d1;i++){
    for (int j=0;j<d2;j++){
      delete[] ptr[i][j];
    }
    delete[] ptr[i];
  }
  delete[] ptr;
}


template<typename Ttype>
void allocate4D(Ttype****& ptr, int d1, int d2, int d3, int d4){
  ptr = new Ttype***[d1];
  for (int i=0;i<d1;i++){
    ptr[i] = new Ttype**[d2];
    for (int j=0;j<d2;j++){
      ptr[i][j] = new Ttype*[d3];
      for(int k=0;k<d3;k++){
        ptr[i][j][k] = new Ttype[d4];
        for(int l=0;l<d4;l++){
          ptr[i][j][k][l] = Ttype(0);
        }
      }
    }
  }


  /*
    ptr = new Ttype***[d1];
    for(int k=0;k<d1;k++) allocate3D<Ttype>(ptr[k], d2, d3, d4);
  */
}
template<typename Ttype>
void deallocate4D(Ttype****& ptr, int d1, int d2, int d3, int d4){
  for (int i=0;i<d1;i++){
    for (int j=0;j<d2;j++){
      for(int k=0;k<d3;k++){
	delete[] ptr[i][j][k];
      }
      delete[] ptr[i][j];
    }
    delete[] ptr[i];
  }
  delete[] ptr;
}


#endif
