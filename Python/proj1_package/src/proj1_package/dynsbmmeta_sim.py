import numpy as np
from scipy.stats import norm, poisson, nbinom, bernoulli, multivariate_normal, multinomial, uniform


def gen_trans_mat(p_stay,Q):
    """
    Generate simple transition matrix with fixed probability of group persistence, else uniform random choice of remaining groups

    Args:
        p_stay (float): Group persistence probability
        Q (int): Number of groups

    Returns:
        [type]: [description]
    """
    return np.identity(Q)*(p_stay+(p_stay-1)/(Q-1))+np.ones((Q,Q))*(1-p_stay)/(Q-1)

def gen_intergroup_probs(p_in,p_out,Q):
    # actually not even necessary, would only be worth producing if intergroup probs were more different
    return np.identity(Q)*(p_in-p_out) + np.ones((Q,Q))*p_out

def gen_beta_mat(Q,p_in,p_out):
    return (p_in-p_out)*np.identity(Q)+p_out*np.ones((Q,Q))

def gen_ppm(Z,p_in=0.4,p_out=0.1,beta_mat=None,self_loops=False):
    """
    Generate planted partition matrix given partition, and group edge probabilities

    Args:
        Z ([type]): [description]
        p_in (float, optional): [description]. Defaults to 0.4.
        p_out (float, optional): [description]. Defaults to 0.1.
        beta_mat ([type], optional): [description]. Defaults to None.
        self_loops (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    # K=len(sizes)
    K = Z.max()+1
    N=len(Z)
    A=np.random.rand(N,N)
    # cum_size=np.cumsum(sizes)
    if beta_mat is None:
        for i in range(N):
            for j in range(i,N):
                # need to zero opposite value
                if j!=i:
                    A[j,i]=0.0
                if Z[i]==Z[j]:
                    A[i,j]=(A[i,j]<=p_in)*1.
                else:
                    A[i,j]=(A[i,j]<=p_out)*1.
        # # Can't do below, as need to use Zs!! TODO: Fix with more efficient implementation used for metadata below
        # for k in range(K):
        #     for l in range(K):
        #         if k==0:
        #             if l==0:
        #                 A[:sizes[0],:sizes[0]]=(A[:sizes[0],:sizes[0]]<=p_in)*1.
        #             else:
        #                 A[:sizes[0],cum_size[l-1]:cum_size[l]]=(A[:sizes[0],cum_size[l-1]:cum_size[l]]<=p_out)*1.
        #         else:
        #             if l==0:
        #                 A[cum_size[k-1]:cum_size[k],:sizes[0]]=(A[cum_size[k-1]:cum_size[k],:sizes[0]]<=p_out)*1.
        #             else:
        #                 if k==l:
        #                     A[cum_size[k-1]:cum_size[k],cum_size[l-1]:cum_size[l]]=(A[cum_size[k-1]:cum_size[k],cum_size[l-1]:cum_size[l]]<=p_in)*1.
        #                 else:
        #                     A[cum_size[k-1]:cum_size[k],cum_size[l-1]:cum_size[l]]=(A[cum_size[k-1]:cum_size[k],cum_size[l-1]:cum_size[l]]<=p_out)*1.
    else: 
        for i in range(N):
            for j in range(i,N):
                if j!=i:
                    A[j,i]=0.
                A[i,j]=(A[i,j]<=beta_mat[Z[i],Z[j]])*1.
        # for k in range(K):
        #     for l in range(K):
        #         if k==0:
        #             if l==0:
        #                 A[:sizes[0],:sizes[0]]=(A[:sizes[0],:sizes[0]]<=beta_mat[0,0])*1.
        #             else:
        #                 A[:sizes[0],cum_size[l-1]:cum_size[l]]=(A[:sizes[0],cum_size[l-1]:cum_size[l]]<=beta_mat[0,l])*1.
        #         else:
        #             if l==0:
        #                 A[cum_size[k-1]:cum_size[k],:sizes[0]]=(A[cum_size[k-1]:cum_size[k],:sizes[0]]<=beta_mat[k,0])*1.
        #             else:
        #                 A[cum_size[k-1]:cum_size[k],cum_size[l-1]:cum_size[l]]=(A[cum_size[k-1]:cum_size[k],cum_size[l-1]:cum_size[l]]<=beta_mat[k,l])*1.


    # A = (((A+A.T)/2)>0)*1.
    A = A + A.T
    if not self_loops:
        for i in range(N):
            A[i,i]=0.
    return A

def pick_category(p_dist,n_samps):
    if n_samps>1:
        return np.random.choice(len(p_dist),size=n_samps,p=p_dist)
    else:
        return np.random.choice(len(p_dist),size=1,p=p_dist)[0]

def evolve_Z(Z_1,trans_prob,T):
    Z = np.zeros((len(Z_1),T))
    Z[:,0]=Z_1
    for i in range(T-1):
        Z[:,i+1] = np.array([pick_category(trans_prob[int(zi),:],1) for zi in Z[:,i]])
    return Z.astype(np.int32)

# print(evolve_Z(np.random.randint(0,high=4,size=(4,)),gen_trans_mat(0.5,4),5))

def get_edge_prob(zi,zj,p_in,p_out):
    if zi==zj:
        return p_in
    else:
        return p_out

def sample_dynsbm_meta(Z_1=np.zeros((10,)),Q=10,T=10,meta_types=['normal','poisson','nbinom','indep bernoulli','categorical'],
                       meta_dims=[],trans_prob=gen_trans_mat(0.8,10),meta_params=[],p_in=0.5,p_out=0.1,beta_mat=None,meta_part=None,
                       ZTP_params=None):
    # Sample a suitable simulated network for the dynamic SBM with metadata
    # intergroup_probs=gen_intergroup_probs(0.5,0.2,10)
    N=len(Z_1)
    S=len(meta_types)
    
    # generate Z
    Z = evolve_Z(Z_1,trans_prob,T)
    
    # Done metadata correctly and intelligently...
    sizes=np.array([[len([i for i in Z[:,t] if i==q]) for q in range(Q)] for t in range(T)]).T
    if meta_part is None:
        meta_sizes=sizes
    else:
        meta_sizes=np.array([[len([i for i in meta_part[:,t] if i==q]) for q in range(Q)] for t in range(T)]).T
    
    # generate metadata
    Xt = {metatype:np.zeros((meta_dims[i],N,T)) for i,metatype in enumerate(meta_types)}
    # params in Ds x Q x T shape - require 3d array even if Ds==1
    for i,meta_type in enumerate(meta_types):
        params=meta_params[i]
        # print(params)
    
        if meta_type=='normal':
            # passing mean and sd
            # initially assuming just 1d normal, generalise later
            X=[[norm.rvs(loc=params[0,q,t],scale=params[1,q,t],size=(meta_sizes[q,t],)) for q in range(Q)] for t in range(T)]

        elif meta_type=='poisson':
            # passing lambda (mean)
            X=[[poisson.rvs(params[0,q,t],size=(meta_sizes[q,t],)) for q in range(Q)] for t in range(T)]
            # print('Poisson: ',len(X))

        elif meta_type=='nbinom':
            # passing r and p
            X=[[nbinom.rvs(params[0,q,t],params[1,q,t],size=(meta_sizes[q,t],)) for q in range(Q)] for t in range(T)]

        elif meta_type=='indep bernoulli':
            # passing independent probabilities of each category 
            # means generating L x |Zq| x Q x T array - check
            L=len(params[:,0,0])
            X=[[np.array([bernoulli.rvs(params[l,q,t],size=(meta_sizes[q,t],)) for l in range(L)]) for q in range(Q)] for t in range(T)]
            # print('Bernoulli: ',X.shape)

        elif meta_type=='categorical':
            # passing distribution over categories
            X=[[pick_category(params[:,q,t],meta_sizes[q,t]) for q in range(Q)] for t in range(T)]

        else:
            raise ValueError(f"Unrecognised metadata distribution: {meta_type}")
        idxs={}
        for q in range(Q):
            if meta_part is None:
                idxs[q]=Z==q
            else:
                idxs[q]=meta_part==q
            for t in range(T):
                if meta_dims[i]==1:
                    Xt[meta_type][0,idxs[q][:,t],t]=X[t][q]
                else:
                    Xt[meta_type][:,idxs[q][:,t],t]=X[t][q]
                

    
    # generate networks
    # inefficient, could sample total number of edges between each pair of groups, then randomly distribute these to each pair of nodes as no degree correction, or sample blocks according to size then put back together (probably best)
    # yes - obviously best thing to do is create matrix of all edge probs (easy as can construct blockwise), sample uniform randomly full matrix, then allow edge if sample <= edge prob - fine as only binary here anyway
    if beta_mat is None:
        A = np.array([gen_ppm(Z[:,t],p_in,p_out) for t in range(T)])
    else:
        A = np.array([gen_ppm(Z[:,t],beta_mat=beta_mat) for t in range(T)])
    if ZTP_params is not None:
        # Assume passing a T x Q x Q array of Poisson means
        idxs=[[Z[:,t]==q for q in range(Q)] for t in range(T)]
        A_zeros = A==0
        for t in range(T):
            for q in range(Q):
                for r in range(Q):
                    pois_tmp = poisson.rvs(ZTP_params[t,q,r],size=(idxs[t][q].sum(),idxs[t][r].sum()))
                    pois_zeros = pois_tmp==0
                    # ensure all values >0
                    while pois_zeros.sum()>0:
                        pois_tmp[pois_zeros] = poisson.rvs(ZTP_params[t,q,r],size=pois_zeros.sum())
                        pois_zeros=pois_tmp==0
                    A[t][np.ix_(idxs[t][q],idxs[t][r])]=pois_tmp
        if A_zeros.sum()==0:
            print("Problems")
        A[A_zeros]=0.0
        
        

    return {'A':A,'X':Xt,'Z':Z,'sizes':sizes}


