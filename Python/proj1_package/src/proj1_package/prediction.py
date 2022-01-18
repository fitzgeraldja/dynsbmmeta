import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt
plt.style.use('/home/fitzgeraldj/Documents/main_project/confirmation/code/dynsbm_meta/python/proj1_package/src/proj1_package/proj1_7clrs.mplstyle')

# example code for testing colors
# for k in range(4,10):
#     cmap=plt.cm.get_cmap('tab20b',k)
#     fig,ax=plt.subplots()
#     x=np.linspace(0,1,10)
#     for j in range(1,7):
#         ax.plot(x,j*np.ones_like(x),color=cmap(j-1))
#     ax.set_title(f"k is {k}")
#     fig.show()


def mask_network(A,mask_percent=5):
    """
    Mask mask_percent of input TxNxN temporal adjacency
    matrix A within each timeslice -- consider edges
    and non-edges separately to avoid class imbalance
    issues.
    
    Not yet usable

    Args:
        A (T x N x N np.array): Temporal adjacency matrix
        mask_percent (int, optional): Percent of edges and non-edges
                                      to mask. Defaults to 5.
    """
    T = A.shape[0]
    mA = np.zeros_like(A)
    edges = np.nonzero(A>0)
    sel_edges = np.random.choice(edges,size=len(edges)//(100/mask_percent))
    non_edges = np.nonzero(A==0)
    sel_non_edges = np.random.choice(non_edges,size=len(non_edges)//(100/mask_percent))
    for t in range(T):
        
        mA[t,sel_edges[t]]=np.nan
        mA[t,sel_non_edges[t]]=np.nan
    return mA
        
        
def mask_metadata(X,mask_percent=5):
    """
    Mask mask_percent of input SxDsxTxN temporal metadata
    matrix A within each timeslice -- consider edges
    and non-edges separately to avoid class imbalance
    issues.
    
    Not yet usable

    Args:
        X (S x Ds x T x N np.array): Metadata matrix
        mask_percent (int, optional): Percent of metadata to mask. Defaults to 5.
        
    """
    T = X.shape[2]
    
def zero_network(A,mask_percent_zs=5,mask_percent_nzs=5):
    """
    Zero some percent of edges, and return indices of both edges and non-edges

    Args:
        A ([type]): [description]
        mask_percent_zs (int, optional): [description]. Defaults to 5.
        mask_percent_nzs (int, optional): [description]. Defaults to 5.
    """
    new_A = A.copy()
    A_zs = A==0
    sz_zs = A_zs.sum(axis=(1,2))
    A_nzs = A!=0
    sz_nzs = A_nzs.sum(axis=(1,2))
    rng = np.random.default_rng()
    z_idxs = [rng.choice(np.array(np.nonzero(A_zs[t])).T,replace=False,size=int(sz_zs[t]//(100/mask_percent_zs)),axis=0).T for t in range(A.shape[0])]
    nz_idxs = [rng.choice(np.array(np.nonzero(A_nzs[t])).T,replace=False,size=int(sz_nzs[t]//(100/mask_percent_nzs)),axis=0).T for t in range(A.shape[0])]
    for t in range(A.shape[0]):
        new_A[t,nz_idxs[t][0,:],nz_idxs[t][1,:]]=0.0  
    return z_idxs,nz_idxs,new_A
    
    
def add_noise_network(A,p_10=0.05,p_01=0.01):
    """
    Add binary noise to a network -- turn edges off with probability p_10, and on with probability p_01

    Args:
        A ([type]): [description]
        p_10 (float, optional): [description]. Defaults to 0.05.
        p_01 (float, optional): [description]. Defaults to 0.01.

    Returns:
        [type]: [description]
    """
    A_zs = A==0
    A_nzs = A!=0
    rA = np.random.rand(*A.shape)
    A[A_zs&(rA<p_01)]=1.0
    A[A_nzs&(rA<p_10)]=0.0
    idxs_01 = np.nonzero(A_zs&(rA<p_01))
    idxs_10 = np.nonzero(A_nzs&(rA<p_10))
    return idxs_01,idxs_10,A

def time_shuffle(A):
    """
    Assume passes T x ... temporal matrix. Returns shuffled matrix and indices to reproduce in other matrices (i.e. metadata).
    Use for temporal bootstrapping comparisons - e.g. want likelihood of our model to be greater than (1-alpha) proportion of sampled models

    Args:
        A ([type]): [description]
    
    Returns: 
        [type]: [description]
    """
    idxs = np.random.permutation(A.shape[0])
    return idxs, A[idxs]

def erase_topology(A):
    """
    Draw from configuration model with same degrees in each timeslice as given matrix

    Args:
        A ([type]): [description]
        
     Returns: 
        [type]: [description]
    """
    
def mask_metadata(X,ps=[]):
    """
    Mask metadata according to list of probs, and return masked metadata along with mask locations

    Args:
        X ([type]): [description]
        ps (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """
    S = len(X)
    if len(ps)!=S:
        raise ValueError("Incorrect number of metadata masking probs passed")
    nX = []
    mask_locs = []
    for s,p in enumerate(ps):
        rX = np.random.rand(*X[s].shape[:2])
        mask_loc = np.argwhere(rX<p)
        nXs = X[s].copy().astype(float)
        nXs[rX<p]=np.nan
        nX.append(nXs)
        mask_locs.append(mask_loc)
    
    return mask_locs,nX

def perturb_metadata(X,metatypes=[],ps=[]):
    """
    Add noise to metadata, and return along with indices of perturbed metadata.

    Args:
        X ([type]): [description]
        metatypes (list, optional): [description]. Defaults to [].
        ps (list, optional): [description]. Defaults to [].
    """
    S = len(X)
    if len(ps)!=S or len(metatypes)!=S:
        raise ValueError("Incorrect number of metadata types/probs passed")
    nX = []
    mask_locs = []
    for s,p in enumerate(ps):
        rX = np.random.rand(*X[s].shape[:2])
        mask_loc = np.argwhere(rX<p)
        nXs = X[s].copy()
        if 'categorical' in metatypes[s] or 'bernoulli' in metatypes[s]:
            L = int(metatypes[s].split()[-1])
            new_meta = np.random.choice(L,size=rX[rX<p].shape) 
            new_meta = pd.get_dummies(new_meta)
            nXs[rX<p] = new_meta
        elif metatypes[s]=='poisson':
            nXs[rX<p] = X[s][rX<p] + np.random.randint(-3,high=3,size=X[s][rX<p])
            nXs[nXs<0] = 0 
        nX.append(nXs)
        mask_locs.append(mask_loc)
    return mask_locs,nX
                      


def predict_links(dynsbmmeta,top_k=None, edges=None, nodes=None, metadata=None, taum=None):
    """
    Predict specified links/non-links given model.
    
    At simplest level, likelihood of edges between two nodes is solely dependent on their group, hence we can immediately calculate this. 
    Of course this is not the same as ranking all possible edges and seeing if those with highest probability exist.
    If predicting all edges incident upon a node simultaneously, then can use metadata if provided to help guide the selection of the group.
    
    Note if we have set of models, we can modify the probability accordingly, i.e. combine the forecasts as e.g. here https://www.sciencedirect.com/science/article/pii/S0169207013001635 
    Simplest way to do so is take the mean, though could also take logistic function of mean log odds etc.
    
    In fact, we also have estimates of marginal probability of the node belonging to each group, hence we can approximate a Bayesian method,
    i.e. P(A_ij)=\sum_{q,r} p(A_ij | z_i=q, z_j=r)p(z_i = q)p(z_j = r).
    
    TODO: Modify to allow passing multiple models, and associated model weights to perform model averaging (e.g. by AIC_weight or otherwise)

    Args:
        dynsbmmeta ([type]): [description]
        top_k (int, optional): [description]. Defaults to None.
        edges (m x 3 np.array, optional): m edges provided, where each row contains t,i,j indices. Defaults to None.
        metadata ([type]): [description]. Defaults to None.
        taum ([type]): [description]. Defaults to None.
    """
    if edges is not None:
        # Performing prediction over specified edges
        if top_k is None:
            if taum is None:
                # calculating raw likelihood of edge (i.e.) solely using groups of nodes, assuming both nodes are present in the sampled network
                if len(dynsbmmeta.beta_mat.shape)==2:
                    probs = np.array([dynsbmmeta.beta_mat[dynsbmmeta.Z[t,i],dynsbmmeta.Z[t,j]] for t,i,j in edges])
                elif len(dynsbmmeta.beta_mat.shape)==3:
                    probs = np.array([dynsbmmeta.beta_mat[t,dynsbmmeta.Z[t,i],dynsbmmeta.Z[t,j]] for t,i,j in edges])
                # just return probs so can test optimal threshold - e.g. average likelihood of edge rather than 0.5
                return probs 
            else:
                # use estimates of node marginals to guide likelihood of edge 
                if len(dynsbmmeta.beta_mat.shape)==2:
                    probs = np.array([np.sum([[taum[t,i,q]*taum[t,j,r]*dynsbmmeta.beta_mat[q,r] for r in range(dynsbmmeta.Q)] for q in range(dynsbmmeta.Q)]) 
                                    for t,i,j in edges])
                elif len(dynsbmmeta.beta_mat.shape)==3:
                    probs = np.array([np.sum([[taum[t,i,q]*taum[t,j,r]*dynsbmmeta.beta_mat[t,q,r] for r in range(dynsbmmeta.Q)] for q in range(dynsbmmeta.Q)],axis=None) 
                                    for t,i,j in edges])
                return probs
            
        else:
            # assume given only edges that exist, and want to see if these are among top_k most likely - NB using scipy to handle ties, so slower than numpy option
            # claiming success if edge in top_k ranks, though note as not degree corrected these are not distinguishable between blocks: as such there are only Q^2 possibilities
            print("Using top k evaluation for edges: note only Q^2 possible ranks (for directed case)\nor Q*(Q+1)/2 for undirected models")
            true_in_topk=np.zeros((len(edges),))
            for m,(t,i,j) in enumerate(edges):
                probs = np.array([dynsbmmeta.beta_mat[dynsbmmeta.Z[t,i],dynsbmmeta.Z[t,k]] for k in range(dynsbmmeta.N)])
                ranks = rankdata(probs,method='min')
                if ranks[j]<top_k:
                    true_in_topk[m]=1
            return true_in_topk
    if nodes is not None:
        # assume calculating proportion of top_k predictions for each node that actually exist? problem of breaking ties
        
        # assume calculating accuracy of placing node in group according to metadata?
        pass          
                

def calc_roc_auc(probs,true,ax=None,plot=True):
    """
    Generate ROC plot for link prediction in binary network case

    Args:
        probs ([type]): Estimated probabilities of links
        true ([type]): Actual link values (existence = 1)
        # preds ([type]): Predicted values
    """
    # # get idxs to sort - note ascending sort, so do minus array to get in descending
    # idxs = np.argsort(-probs)
    # true = true[idxs]
    # preds = preds[idxs]
    # pos_count = true.sum()
    # neg_count = len(true)-pos_count
    # if pos_count==0:
    #     raise ValueError("Invalid data passed: no true positives")
    # elif neg_count==0:
    #     raise ValueError("Invalid data passed: no true negatives")
    # # calc x axis vals - this is 1 - specificity, which is the false positive fraction = FP/(FP+TN)
    # x = np.cumsum((true==0)&(preds==1))/(neg_count)
    # x = np.concatenate([x,[1]])
    
    # # calc y axis vals - this is sensitivity, which is true positive fraction = TP/(TP+FN)
    # y = np.cumsum((true==1)&(preds==1))/pos_count 
    # y = np.concatenate([y,[1]])
    
    fpr,tpr,_ = roc_curve(true,probs)
    roc_auc = auc(fpr,tpr) 
    
    if plot:
        if ax is None:
            fig,new_ax = plt.subplots()
        else:
            new_ax = ax
        new_ax.plot(fpr,tpr)
        new_ax.set_xlabel('False positive rate')
        new_ax.set_ylabel('True positive rate')
        new_ax.plot([0,1],[0,1],linewidth=2.0,linestyle='--',color='grey',label='_nolegend_')
        if ax is None:
            fig.show()
    return roc_auc


def predict_metadata(dynsbmmeta,nodes=None,taum=None):
    """
    Much as for links, we can infer metadata through either directly taking the most likely value from inferred parameters (i.e. group mean),
    or use the approximated node marginals to further tune results.

    Args:
        dynsbmmeta ([type]): [description]
        nodes (Ns x 2 np.array, optional): List of (time,node_idx) to infer metadata for. Defaults to None.
        taum ([type], optional): [description]. Defaults to None.
    """ 
    if taum is None:
        metatypes = dynsbmmeta.metatypes
        metaparams = dynsbmmeta.metaparams
        expected_X = {}
        for node in nodes:
            t,nidx = node
            expected_X[(t,nidx)]=[]
            for s,metatype in enumerate(metatypes):
                if metatype == 'poisson':
                    expected_X[(t,nidx)].append(np.floor(metaparams[s][t,dynsbmmeta.Z[t,nidx],0]))
                elif 'bernoulli' in metatype or 'categorical' in metatype:
                    expected_X[(t,nidx)].append(np.argmax(metaparams[s][t,dynsbmmeta.Z[t,nidx],:]))
        return expected_X  
    else:
        metatypes = dynsbmmeta.metatypes
        metaparams = dynsbmmeta.metaparams
        expected_X = {}
        for node in nodes:
            t,nidx = node
            expected_X[(t,nidx)]=[]
            for s,metatype in enumerate(metatypes):
                if metatype == 'poisson':
                    expected_X[(t,nidx)].append(np.floor(np.sum([metaparams[s][t,q,0]*taum[t,nidx,q] for q in range(dynsbmmeta.Q)])))
                elif 'bernoulli' in metatype or 'categorical' in metatype:
                    expected_X[(t,nidx)].append(np.argmax(np.sum([taum[t,nidx,q]*metaparams[s][t,q,:] for q in range(dynsbmmeta.Q)],axis=0)))
        return expected_X  
        
    
    