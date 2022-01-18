import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import norm, poisson, nbinom, bernoulli, multivariate_normal, multinomial, chisquare, entropy, wasserstein_distance
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.neighbors import KernelDensity
# make plots prettier
plt.rcParams.update({
    "text.usetex":True,
    "font.family":"sans-serif" # or "serif" or "cursive" or "monospace"
    # "font.sans-serif":["Helvetica" or "Avant Garde" or "Computer Modern Sans Serif"]
    # "font.serif":["Times" or "Palatino" or "New Century Schoolbook" or "Bookman" or "Computer Modern Roman"]
    # Computer modern fonts default
})
# fns for working w r data
import rpy2.robjects as robjects
# fns for rmi
from math import log,exp
from scipy.special import gammaln

# for limiting nb cores used incl in sklearn
# from threadpoolctl import ThreadpoolController

# controller = ThreadpoolController()
# with controller(limits=n_cores,
#                 user_api='blas',
#                 # user_api='openmp',
#                 ):

# for sankey plots
from matplotlib.colors import to_hex
import plotly.graph_objects as go
from collections import Counter


class Dynsbmmeta_base(object):

    def __init__(self,X=None,A=None,Q=3,Z=None,beta_mat=None,trans_mat=None,metatypes=None,metaparams=None,metadists=None) -> None:
        """
        Initialise base dynsbmmeta class

        Args:
            X ([type], optional): [description]. Defaults to None.
            A ([type], optional): [description]. Defaults to None.
            Q (int, optional): [description]. Defaults to 3.
            Z ([type], optional): [description]. Defaults to None.
            beta_mat ([type], optional): [description]. Defaults to None.
            trans_mat ([type], optional): [description]. Defaults to None.
            metatypes ([type], optional): [description]. Defaults to None.
            metaparams ([type], optional): [description]. Defaults to None.
            metadists ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        self.X = X # assume in S x T x N x Ds form
        self.A = A # assume in T x N x N form

        if Z is None:
            self.Q = Q
        else:
            self.Q = np.amax(Z)+1
        if A is not None:
            self.T = A.shape[0]
            self.N = A.shape[1]
        elif Z is not None:
            self.T = Z.shape[0]
            self.N = Z.shape[1]
        if beta_mat is not None:
            self.beta_mat = beta_mat
        if trans_mat is not None:
            self.trans_mat = trans_mat  
        self.Z = Z # will return in T x N form
        self.params = None
        self.metatypes = metatypes # assume list length S with string names of metadata type
        self.metaparams = metaparams # assume S x T x Q x Ds form
        if metadists is None and metaparams is not None:
            # distributions of type given for each group with inferred parameters at time T, so access with self.metadists[s][t][q]. Expect of type s.t. can query likelihood of a point with .pdf(x) unless categorical, in which case just access [x]
            self.metadists=[] 
        else:
            self.metadists=metadists
        
        
    def plot_metadists(self,metatype=None,t=None,q=None):
        # Compare data distribution within group to that predicted
        # Take variational parameters in T x Q x Ds form (assume select those for specific piece chosen already)
        
        s=self.metatypes.index(metatype)
        if (metatype=="poisson")|(metatype=="negative binomial"):
            if q is None:
                metadists = self.metadists[s][t]
                for q_ in range(self.Q):
                    nodes = [i for i,x in enumerate(self.Z[t,:]) if x==q_]
                    metadata = self.X[s,t,nodes,:]
                    
                    fig,ax=plt.subplots(dpi=300)
                    ax.set_title(f"Metadata for {metatype} at time {t}, group {q_}")
                    
                    x=np.linspace(metadists[q_].ppf(0.001),metadists[q_].ppf(0.999),num=100)
                    ax.plot(x,metadists[q_].pdf(x),)
                    ax.hist(metadata,density=True,alpha=0.2,histtype="stepfilled")
                    fig.show()
            else:
                metadist = self.metadists[s][t][q]
                # distname = type(metadist.dist).__name__
                nodes = [i for i,x in enumerate(self.Z[t,:]) if x==q]
                metadata = self.X[s,t,nodes,:]
                
                fig,ax=plt.subplots(dpi=300)
                ax.set_title(f"Metadata for {metatype} at time {t}, group {q}")
                x=np.linspace(metadist.ppf(0.001),metadist.ppf(0.999),num=100)
                ax.plot(x,metadist.pdf(x),)
                ax.hist(metadata,density=True,alpha=0.2,histtype="stepfilled")
                fig.show()
        elif ("bernoulli" in metatype)|("categorical" in metatype):
            # dealing with some kind of categorical data - do bar charts
            # assuming one/k-hot metadata
            L=len(self.X[s,t,0,:])
            if q is None:
                metadists = self.metadists[s][t]
                for q_ in range(self.Q):
                    nodes = [i for i,x in enumerate(self.Z[t,:]) if x==q_]
                    metadata = self.X[s,t,nodes,:]
                    fig,ax=plt.subplots(dpi=300)
                    ax.set_title(f"Metadata for {metatype} at time {t}, group {q_}")
                    
                    x = np.arange(L)
                    width = 0.35
                    rects1 = ax.bar(x - width/2,[metadists[q_][x_] for x_ in x],width,label='Estimate')
                    rects2 = ax.bar(x+width/2,float(np.sum(metadata,axis=1))/len(metadata[:,0]),width,label='Observed')
                    
                    ax.set_ylabel('Categories')
                    ax.set_xticks(x)
                    ax.set_xticklabels(x)
                    ax.legend()

                    ax.bar_label(rects1,padding=3)
                    ax.bar_label(rects2,padding=3)

                    fig.tight_layout()
                    fig.show()

            else:
                metadist = self.metadists[s][t][q]
                # distname = type(metadist.dist).__name__
                nodes = [i for i,x in enumerate(self.Z[t,:]) if x==q]
                metadata = self.X[s,t,nodes,:]
                
                fig,ax=plt.subplots(dpi=300)
                ax.set_title(f"Metadata for {metatype} at time {t}, group {q}")
                x = np.arange(L)
                width = 0.35
                rects1 = ax.bar(x - width/2,[metadist[x_] for x_ in x],width,label='Estimate')
                rects2 = ax.bar(x+width/2,float(np.sum(metadata,axis=1))/len(metadata[:,0]),width,label='Observed')
                
                ax.set_ylabel('Categories')
                ax.set_xticks(x)
                ax.set_xticklabels(x)
                ax.legend()

                ax.bar_label(rects1,padding=3)
                ax.bar_label(rects2,padding=3)

                fig.tight_layout()
                fig.show()

        else:
            # dealing with multivariate normal data - project into 2d first, then do heat maps
            if q is None:
                metadists = self.metadists[s][t]
                for q_ in range(self.Q):
                    nodes = [i for i,x in enumerate(self.Z[t,:]) if x==q]
                    pca = PCA(n_components=2).fit(self.X[s,t,nodes,:])
                    proj_data = pca.transform(self.X[s,t,nodes,:])
                    
                    fig,ax=plt.subplots(dpi=300)
                    ax.hist2d(proj_data[:,0],proj_data[:,1],density=True)

                    deltaX = (max(proj_data[:,0])-min(proj_data[:,0]))/10
                    deltaY = (max(proj_data[:,1])-min(proj_data[:,1]))/10
                    xmin = min(proj_data[:,0]) - deltaX
                    xmax = max(proj_data[:,0]) + deltaX
                    ymin = min(proj_data[:,1]) - deltaY
                    ymax = max(proj_data[:,1]) + deltaX
                    # create meshgrid
                    xx,yy = np.mgrid[xmin:xmax:100j,ymin:ymax:100j]
                    posns = np.column_stack((xx.ravel(),yy.ravel()))
                    f = np.reshape(metadists[q_].pdf(pca.inverse_transform(posns)),xx.shape)
                    cset = ax.contour(xx,yy,f,colors='w')
                    ax.clabel(cset,inline=1,fontsize=10,color='w')
                    ax.set_xlabel("PCA_1")
                    ax.set_ylabel("PCA_2")
                    ax.set_title(f"Metadata for {metatype} at time {t}, group {q_}")
                    fig.tight_layout()
                    fig.show()

            else:
                metadist = self.metadists[s][t][q]

    # Implement KDE plots?

    def kl_div_dist(self,metatype=None,t=0,q=0,true=None):
        s=self.metatypes.index(metatype)
        # nodes = [i for i,x in enumerate(self.Z[t,:]) if x==q]
        # metadata,bins=np.histogram(self.X[s,t,nodes,:],bins=10,density=True)
        # Don't need this^ we want distance from true distribution, which we know for sim data
        metadist=self.metadists[s][t][q]
        vals=metadist.ppf([0.001,0.999])
        x=np.linspace(vals[0],vals[1],num=100)
        # assume true has pdf attribute
        return entropy(metadist.pdf(x),qk=true.pdf(x))
    
    def emd(self,metatype=None,t=0,q=0,true=None):
        # also known as wasserstein distance (w particular power)
        s=self.metatypes.index(metatype)
        metadist=self.metadists[s][t][q]
        vals = metadist.ppf([0.001,0.999])
        x=np.linspace(vals[0],vals[1],num=100)
        # assume true has pdf attribute
        return wasserstein_distance(metadist.pdf(x),true.pdf(x))





    # Create models from data
    def best_fit_distribution(self,data, bins=200, ax=None):
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Best holders
        best_distributions = []

        # Estimate distribution parameters from data
        for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

            print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

            distribution = getattr(st, distribution)

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    
                    # if axis pass in add to plot
                    try:
                        if ax:
                            pd.Series(pdf, x).plot(ax=ax)
                        pass
                    except Exception:
                        pass

                    # identify if this distribution is better
                    best_distributions.append((distribution, params, sse))
            
            except Exception:
                pass

        # so best_dist=result[0], with best_dist[0]=dist,best_dist[1]=params (pass these to make_pdf below)
        return sorted(best_distributions, key=lambda x:x[2])

    def make_pdf(self,dist, params, size=10000):
        """Generate distributions's Probability Distribution Function """

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)

        return pdf

    def chisq_test(self,metatype=None,t=0,q=0):
        # suitable for comparing discrete distributions
        # compare chi2 stat to chi2 dist with k-p-1 dof, with k the number of bins, and p the number of params
        # inferred
        # problem is binning the data 
        # want minimum frequency of 5 in each bin, and 
        s=self.metatypes.index(metatype)
        nodes = [i for i,x in enumerate(self.Z[t,:]) if x==q]
        metadata = self.X[s,t,nodes,:]
        val_counts = Counter(metadata)
        chisq,p_val=chisquare(val_counts.values(),f_exp=len(metadata)*np.array([self.metadist[s][t][q].pdf(x) for x in val_counts.keys()]),ddof=len(self.metaparams[s][t][q]))
        return chisq,p_val

            

    def gen_trans_mat(self,p_stay,Q):
        return np.identity(Q)*(p_stay+(p_stay-1)/(Q-1))+np.ones((Q,Q))*(1-p_stay)/(Q-1)

    def gen_intergroup_probs(self,p_in,p_out,Q):
        # actually not even necessary, would only be worth producing if intergroup probs were more different
        return np.identity(Q)*(p_in-p_out) + np.ones((Q,Q))*p_out

    def gen_ppm(self,Z,sizes,p_in,p_out,self_loops=False):
        """
        NOTE incorrectly implemented - need to take partition as argument then correctly locate edges, rather than solely use group sizes. See metadata generation process
        for more efficient implementation of how this could be done, or sim file for inefficient. Would want to generate edges according to sizes of blocks, then distribute accordingly

        Args:
            Z ([type]): [description]
            sizes ([type]): [description]
            p_in ([type]): [description]
            p_out ([type]): [description]
            self_loops (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        K=len(sizes)
        N=sum(sizes)
        A=np.random.rand(N,N)
        cum_size=np.cumsum(sizes)
        for k in range(K):
            for l in range(K):
                if k==0:
                    if l==0:
                        A[:sizes[0],:sizes[0]]=(A[:sizes[0],:sizes[0]]<=p_in)*1.
                    else:
                        A[:sizes[0],cum_size[l-1]:cum_size[l]]=(A[:sizes[0],cum_size[l-1]:cum_size[l]]<=p_out)*1.
                else:
                    if l==0:
                        A[cum_size[k-1]:cum_size[k],:sizes[0]]=(A[cum_size[k-1]:cum_size[k],:sizes[0]]<=p_out)*1.
                    else:
                        if k==l:
                            A[cum_size[k-1]:cum_size[k],cum_size[l-1]:cum_size[l]]=(A[cum_size[k-1]:cum_size[k],cum_size[l-1]:cum_size[l]]<=p_in)*1.
                        else:
                            A[cum_size[k-1]:cum_size[k],cum_size[l-1]:cum_size[l]]=(A[cum_size[k-1]:cum_size[k],cum_size[l-1]:cum_size[l]]<=p_out)*1.
        A = (((A+A.T)/2)>0)*1.
        if not self_loops:
            for i in range(N):
                A[i,i]=0.
        return A

    def pick_category(self,p_dist,n_samps):
        if n_samps>1:
            return np.random.choice(len(p_dist),size=n_samps,p=p_dist)
        else:
            return np.random.choice(len(p_dist),size=1,p=p_dist)[0]

    
    def evolve_Z(self,Z_1,trans_prob,T):
        Z = np.zeros((len(Z_1),T))
        Z[:,0]=Z_1
        for i in range(T-1):
            Z[:,i+1] = np.array([self.pick_category(trans_prob[int(zi),:],1) for zi in Z[:,i]])
        return Z

    # print(evolve_Z(np.random.randint(0,high=4,size=(4,)),gen_trans_mat(0.5,4),5))

    
    def get_edge_prob(self,zi,zj,p_in,p_out):
        if zi==zj:
            return p_in
        else:
            return p_out

    def sample_dynsbm_meta(self,Z_1=np.zeros((10,)),Q=10,T=10,meta_types=['normal','poisson','nbinom','indep bernoulli','categorical'],meta_dims=[],trans_prob=None,meta_params=[],p_in=0.5,p_out=0.1,p_stay=0.8):
        # Sample a suitable simulated network for the dynamic SBM with metadata
        # intergroup_probs=gen_intergroup_probs(0.5,0.2,10)
        N=len(Z_1)
        S=len(meta_types)
        
        if trans_prob is None:
            trans_prob = self.gen_trans_mat(p_stay,Q)

        # generate Z
        Z = self.evolve_Z(Z_1,trans_prob,T)

        sizes=np.array([[len([i for i in Z[:,t] if i==q]) for q in range(Q)] for t in range(T)]).T
        
        # generate metadata
        Xt = {metatype:np.zeros((meta_dims[i],N,T)) for i,metatype in enumerate(meta_types)}
        # params in Ds x Q x T shape - require 3d array even if Ds==1
        for i,meta_type in enumerate(meta_types):
            params=meta_params[i]
            # print(params)
            if meta_type=='normal':
                # passing mean and sd
                # initially assuming just 1d normal, generalise later
                X=[[norm.rvs(loc=params[0,q,t],scale=params[1,q,t],size=(sizes[q,t],)) for q in range(Q)] for t in range(T)]

            elif meta_type=='poisson':
                # passing lambda (mean)
                X=[[poisson.rvs(params[0,q,t],size=(sizes[q,t],)) for q in range(Q)] for t in range(T)]
                # print('Poisson: ',len(X))

            elif meta_type=='nbinom':
                # passing r and p
                X=[[nbinom.rvs(params[0,q,t],params[1,q,t],size=(sizes[q,t],)) for q in range(Q)] for t in range(T)]

            elif meta_type=='indep bernoulli':
                # passing independent probabilities of each category 
                # means generating L x |Zq| x Q x T array - check
                L=len(params[:,0,0])
                X=[[np.array([bernoulli.rvs(params[l,q,t],size=(sizes[q,t],)) for l in range(L)]) for q in range(Q)] for t in range(T)]
                # print('Bernoulli: ',X.shape)

            elif meta_type=='categorical':
                # passing distribution over categories
                X=[[self.pick_category(params[:,q,t],sizes[q,t]) for q in range(Q)] for t in range(T)]

            else:
                raise ValueError(f"Unrecognised metadata distribution: {meta_type}")
            idxs={}
            for q in range(Q):
                idxs[q]=Z==q
                for t in range(T):
                    if meta_dims[i]==1:
                        Xt[meta_type][0,idxs[q][:,t],t]=X[t][q]
                    else:
                        Xt[meta_type][:,idxs[q][:,t],t]=X[t][q]
                    

        
        # generate networks
        # inefficient, could sample total number of edges between each pair of groups, then randomly distribute these to each pair of nodes as no degree correction, or sample blocks according to size then put back together (probably best)
        # yes - obviously best thing to do is create matrix of all edge probs (easy as can construct blockwise), sample uniform randomly full matrix, then allow edge if sample <= edge prob - fine as only binary here anyway
        A = np.array([self.gen_ppm(sizes[:,t],p_in,p_out) for t in range(T)])

        return {'A':A,'X':Xt,'Z':Z,'sizes':sizes}

    def load_r_data(self,path="./",filename=""):
        # load the file
        res = robjects.r['load'](path+filename+".RData")
        return res

    def r_2_np(self,var=""):
        # retrieve the object loaded from file
        a = robjects.r[var]

        # turn into numpy array (assuming vector/matrix/array)
        a = np.array(a)
        return a
    
    def mse(self,param=None,true=None):
        return np.linalg.norm(param - true)
    
    def part_overlap(self,true=None):
        return float(sum(self.Z==true))/self.Z.size


    def rmi(self,true=None,data=None,verbose=False,method='within',show_nmi=False):
        # data shape required is two columns of integer group labels
        # For instance:
        #
        # 0 0
        # 0 0
        # 0 1
        # 1 2
        # 1 0
        # 1 3
        # etc.
        # reshape membership into suitable form if not providing in such already
        if data is None:
            if true is None:
                Exception("Must pass partition to compare with")
            else:
                if method=='total':
                    data = np.column_stack((self.Z.flatten(),true.flatten()))
                    if -1 in np.unique(data):
                        data = data[data!=-1]
                elif method=='within':
                    data = np.stack((self.Z,true)).transpose((1,2,0))
                    # access data in correct form with data[t,:,:] (T x N x 2 (inferred,true))

        if method=='total':
            n = len(data)
            R = len(set(data[:,0]))
            S = len(set(data[:,1]))

            if np.amax(data[:,0])!=R-1:
                # will cause indexing error, need to relabel
                if verbose: print("Inferred: ",*set(data[:,0]))
                _,data[:,0]=np.unique(data[:,0],return_inverse=True)
                if verbose: print("Relabelled: ",*set(data[:,0]))
            # elif np.amax(data[:,1])!=S-1:
            #     print("True:",*set(data[:,1]))

            # Construct the contingency table
            c = np.zeros([R,S],int)
            for k in range(n):
                r,s = data[k]
                r=R-1 if type(r) is type(None) else r # nb this treats non-present nodes as other community
                s=S-1 if type(s) is type(None) else s
        #         if r==R-1:
        #             print(set(data[:,0]),r)
                c[int(r),int(s)] += 1
                    

            a = np.sum(c,axis=1)
            b = np.sum(c,axis=0)


            # Calculate the standard mutual information
            I = gammaln(n+1)
            for r in range(R):
                for s in range(S):
                    I += gammaln(c[r,s]+1)
            for r in range(R): I -= gammaln(a[r]+1)
            for s in range(S): I -= gammaln(b[s]+1)
            I /= (n*log(2))
            
            if show_nmi:
                print(f"Standard NMI: ",nmi(data[:,0],data[:,1]))
                print("MI internal to RMI (in bits): ",I)
            

            # Calculate the correction
            w = n/(n+0.5*R*S)
            x = (1-w)/R + w*a/n
            y = (1-w)/S + w*b/n
            nu = (S+1)/(S*np.sum(np.square(x))) - 1/S
            mu = (R+1)/(R*np.sum(np.square(y))) - 1/R

            logOmega = (R-1)*(S-1)*np.log(n+0.5*R*S) \
                        + 0.5*(R+nu-2)*np.sum(np.log(y)) + 0.5*(S+mu-2)*np.sum(np.log(x)) \
                        + 0.5*(gammaln(mu*R)+gammaln(nu*S) \
                        - R*(gammaln(S)+gammaln(mu)) - S*(gammaln(R)+gammaln(nu)))

            if verbose:
                print("Read",n,"objects with R =",R,"and S =",S)
                print()
                print("Contingency table:")
                print(c)
                print()
                print("Row sums:   ",a)
                print("Column sums:",b)
                print()
                print("Mutual information I =",I,"bits per object")
                print()
                print("Estimated number of contingency tables Omega =",exp(logOmega))
                print()
                # Print the reduced mutual information
                print("Reduced mutual information M =",I-logOmega/(n*log(2)),"bits per object")
            
            return I-logOmega/(n*log(2))
        
        elif method=='within':
            # n = data.shape[1]
            rmis = np.zeros((self.T,1))
            for t in range(self.T):
                datat = data[t,:,:] 
                if -1 in np.unique(datat): 
                    datat = datat[datat!=-1].reshape(((datat!=-1).sum(axis=0)[0],2))
                n = len(datat)
                R = len(set(datat[:,0]))
                S = len(set(datat[:,1]))
                if np.amax(datat[:,0])!=R-1:
                    # will cause indexing error, need to relabel
                    if verbose: print("Inferred: ",*set(datat[:,0]))
                    _,datat[:,0]=np.unique(datat[:,0],return_inverse=True)
                    if verbose: print("Relabelled: ",*set(datat[:,0]))
                if np.amax(datat[:,1])!=S-1:
                    if verbose: print("True:",*set(datat[:,1]))
                    _,datat[:,1]=np.unique(datat[:,1],return_inverse=True)
                    if verbose: print("Relabelled: ",*set(datat[:,1]))

                # Construct the contingency table
                c = np.zeros([R,S],int)
                for k in range(n):
                    r,s = datat[k,:]
                    # r=R-1 if type(r) is type(None) else r # nb this treats non-present nodes as other community
                    # s=S-1 if type(s) is type(None) else s
                    # if r==R-1:
                    #     print(set(data[:,0]),r)
                    c[int(r),int(s)] += 1
                        

                a = np.sum(c,axis=1)
                b = np.sum(c,axis=0)


                # Calculate the standard mutual information
                I = gammaln(n+1)
                for r in range(R):
                    for s in range(S):
                        I += gammaln(c[r,s]+1)
                for r in range(R): I -= gammaln(a[r]+1)
                for s in range(S): I -= gammaln(b[s]+1)
                I /= (n*log(2))
                
                if show_nmi:
                    print(f"Standard NMI at {t}: ",nmi(datat[:,0],datat[:,1]))
                

                # Calculate the correction
                w = n/(n+0.5*R*S)
                x = (1-w)/R + w*a/n
                y = (1-w)/S + w*b/n
                nu = (S+1)/(S*np.sum(np.square(x))) - 1/S
                mu = (R+1)/(R*np.sum(np.square(y))) - 1/R

                logOmega = (R-1)*(S-1)*np.log(n+0.5*R*S) \
                            + 0.5*(R+nu-2)*np.sum(np.log(y)) + 0.5*(S+mu-2)*np.sum(np.log(x)) \
                            + 0.5*(gammaln(mu*R)+gammaln(nu*S) \
                            - R*(gammaln(S)+gammaln(mu)) - S*(gammaln(R)+gammaln(nu)))

                if verbose:
                    print("Read",n,"objects with R =",R,"and S =",S)
                    print()
                    print("Contingency table:")
                    print(c)
                    print()
                    print("Row sums:   ",a)
                    print("Column sums:",b)
                    print()
                    print("Mutual information I =",I,"bits per object")
                    print()
                    print("Estimated number of contingency tables Omega =",exp(logOmega))
                    print()
                    # Print the reduced mutual information
                    print("Reduced mutual information M =",I-logOmega/(n*log(2)),"bits per object")
                
                rmis[t]=I-logOmega/(n*log(2))
            return rmis
            


        

    def nrmi(self,data=None,true=None,method='within',**kwargs):
        if data is None:
            if true is None:
                Exception("Need to provide partition to compare to")
            else:
                if method=='total':
                    data = np.column_stack((self.Z.flatten(),true.flatten()))
                elif method=='within':
                    data = np.stack((self.Z,true)).transpose((1,2,0))
                    # access data in correct form with data[t,:,:] (T x N x 2 (inferred,true))
        if method=='total':    
            redmi=self.rmi(data=data,method=method,**kwargs)
            pi1=self.rmi(data=np.tile(data[:,0],(2,1)).T)
            pi2=self.rmi(data=np.tile(data[:,1],(2,1)).T)
            return 2*redmi/(pi1+pi2)
        elif method=='within':
            redmis=self.rmi(data=data,method=method,**kwargs)
            pi1s=self.rmi(data=np.tile(data[:,:,0],(2,1,1)).transpose((1,2,0)),method=method)
            pi2s=self.rmi(data=np.tile(data[:,:,1],(2,1,1)).transpose((1,2,0)),method=method)
            
            return np.array([2*redmi/(pi1+pi2) for redmi,pi1,pi2 in np.column_stack((redmis,pi1s,pi2s))])

    
    def sankey(self,flow_type='flow',title=None):
        """
        Make Sankey diagram given temporal partition. Note assumed that nodes outside the net are labelled -1.

        Args:
            flow_type (str, optional): Can plot flows by raw number ('raw_flow'), relative flow ('flow), or sized by jaccard index ('jaccard'). Defaults to 'flow'.
            title ([type], optional): [description]. Defaults to None.
        """        """[summary]

        Args:
            flow_type (str, optional): [description]. Defaults to 'flow'.
            title ([type], optional): [description]. Defaults to None.
        """        """[summary]

        Args:
            flow_type (str, optional): [description]. Defaults to 'flow'.
            title ([type], optional): [description]. Defaults to None.
        """        
        
        # NB parts just dictionary of dictionaries, with keys the timestep, then inner key the index of a node, and items the block label found - can just cut directly to
        
        # parts = {t:dict([i,x] for i,x in enumerate(Z[:,t])) for t in ts}
        comms={t:[[i for i,x in enumerate(self.Z[t,:]) if x==q]+[i for i,x in enumerate(self.Z[t,:]) if x==-1] for q in range(self.Q+1)] for t in range(self.T)}

        raw_flows=[]
        flows=[]
        labels=[]
        jaccard=[]
        for t in range(self.T-1):
            for cm in comms[t]:
                raw_flows.append([len(set(cm).intersection(set(com))) for com in comms[t+1]])
                flows.append([len(set(cm).intersection(set(com)))/len(cm) for com in comms[t+1]
                            if len(cm)!=0])
                labels.append([set(cm).intersection(set(com)) for com in comms[t+1]])
                jaccard.append([len(set(cm).intersection(set(com)))/len(set(cm).union(set(com)))
                                if len(set(cm).union(set(com)))!=0 else 0.
                            for com in comms[t+1]])

        # sources=[i for i in range(len([com for cm in list(comms.values())[:-1] for com in cm]))]
        # targets=[i for i in range(6,len([com for cm in list(comms.values()) for com in cm]))]
        ctr=0
        proper_list=[]
        for t in range(self.T-1):
            for i in range(ctr,len(comms[t])+ctr):
                for j in range(len(flows[i])):
                    proper_list.append((i,ctr+len(comms[t])+j,raw_flows[i][j],flows[i][j],repr(labels[i][j]),
                                    jaccard[i][j]))
            ctr+=len(comms[t])
        data=pd.DataFrame(proper_list,columns=['source','target','raw_flow','flow','label','jaccard'])

        
        ctr=0
        outside_net=[]
        for i in range(len(comms)):
            ctr+=len(comms[i])-1
            outside_net.append(ctr)
            ctr+=1

        cmap=plt.cm.get_cmap('tab20b',max(data['target'])
                            )
        all_comms=set(data['source']).union(set(data['target']))
        # rando=np.random.randint(0, high=max(data['target']), size=(len(all_comms),))
        colours=[to_hex(cmap(i % max(data['target'])))
                if i not in outside_net else '#ffffff'
                for i in all_comms]
        delta=0.6 # set jaccard threshold for distinct comms
        for row in data[(1-data['jaccard'])<delta].values:
            source, target, raw_flow, flow, label, jacc = row
            if (source not in outside_net)&(target not in outside_net):
                colours[target]=colours[source]
            # if target < len(colours):
            #     colours[target]=colours[source]
            # else:
            #     print(data)
            #     print(all_comms)
            #     print(len(colours),target)



        fig = go.Figure(data=[go.Sankey(
            node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
        #       label = ["A1", "A2", "B1", "B2", "C1", "C2"],
            color = colours
            ),
            link = dict(
            source = data['source'],
            target = data['target'],
            value = data[flow_type],
            # value = data['jaccard'], # alt
            label = data['label']
        ))])

        fig.update_layout(title_text=title, font_size=12)
        fig.show()

    
class Normaldist(object):
    # class for returning likelihood at different points of multivariate normals of certain classes
    def __init__(self,params,disttype) -> None:
        self.params = params 
        self.__name__ = disttype 
        if "independent" in disttype:
            # product of independent normals
            self.mean = self.params[:len(self.params)/2]
            self.sd = self.params[len(self.params)/2:]
            pass
        elif "shared" in disttype:
            # shared variance parameter
            self.mean = self.params[:-1]
            self.sd = self.params[-1]
            pass
        else:
            # full multivariate normal
            dim = 0.5*(np.sqrt(4*len(self.params)+1)-1)
            self.mean = self.params[:dim]
            self.sd = np.reshape(self.params[dim:],(dim,dim))
            ValueError("Multivariate normal not implemented")
            pass

    def pdf(self,x):
        if "independent" in self.__name__:
            # product of independent normals
            pass
        elif "shared" in self.__name__:
            # shared variance parameter
            pass
        else:
            # full multivariate normal
            pass



        



