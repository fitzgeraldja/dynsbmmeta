from random import sample
import scipy.sparse as sparse
import numpy as np
from sklearn import manifold
from sklearn import neighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm.auto import tqdm
import os 


plt.ion()


##############################################################################
################            LETO CODE BELOW                   ################  
##############################################################################


def calcLL(partitions,edges,LL_fun=None):
    """
    Calculate the log likelihood of a partition - note unchanged, so not incorporating metadata. Also requires undirected network, else breaks - should fix this.

    Args:
        partitions ([type]): [description]
        edges ([type]): [description]
        LL_fun (optional): function with which to modify LL

    Returns:
        LL: n_parts x 2+ np.array, w first column non-DC LL, second DC-LL, and onwards accounting for sampling partitions in some way
    """
    
    if np.min(edges)>0:
        edges-=1
    
    thetas=np.log(10**np.arange(-50,0.1,1)+1e-200)
    oneminusthetas=np.log(1-10**np.arange(-50,0.1,1)+1e-200)
    LL=np.empty((np.shape(partitions)[0],2+len(thetas)))
    n=np.shape(partitions)[1]
    m=np.shape(edges)[0]
    ei=np.arange(m)
    K=len(np.unique(partitions))
    
    for pi,partition in enumerate(partitions):
        nk=np.array([np.sum(partition==k) for k in range(K)])
        mi=partition[edges[:,0]]
        mj=partition[edges[:,1]]
        # print(mi.max(),K)
        A1 = sparse.coo_matrix((np.ones(m),(mi,ei)),shape=(K,m),dtype=np.uint).tocsc()
        A2 = sparse.coo_matrix((np.ones(m),(ei,mj)),shape=(m,K),dtype=np.uint).tocsc()
        
        e_rs = np.array(A1.dot(A2).todense(),dtype=float)
        if not np.all(e_rs==e_rs.T):
            e_rs+=e_rs.T
        nrns = np.outer(nk,nk)
        prs=e_rs/(nrns+1e-200)
        if np.any(np.isnan(np.log(prs+1e-200))):
            raise ValueError("Somehow wrong")
        if np.any(np.isnan(np.log(1-prs+1e-200))):
            print(np.amax(prs))
            raise ValueError("Somehow wrong")
            
        LL[pi,0]= (e_rs*np.log(prs+1e-200)+(nrns-e_rs)*np.log(1-prs+1e-200)).sum()/2.
        
        dr=np.sum(e_rs,0)
        drds=np.outer(dr,dr)
        prs=np.divide(e_rs,drds,where=drds>0)
        prs[drds==0]=1e-10
        LL[pi,1]= (e_rs*np.log(prs+1e-200)).sum()/2. 
        #~ print pi,m,e_rs,e_rs.sum(), LL[pi]
        q=n-np.sum(partitions[0,:]==partition)
        LL[pi,2:]= LL[pi,1] + np.float64([(q)*theta + (n-q)*(oneminustheta) for theta,oneminustheta in zip(thetas,oneminusthetas)]) 
        if LL_fun is not None:
            LL = LL + LL_fun(partition)
    return LL


def calcVI(partitions):
    """
    Calculate a distance matrix based on variation of information. Note could parallelise and significantly speed up...

    Args:
        partitions (n_parts x N np.array): np.array carrying label given in ith partition for jth node at (i,j)

    Returns:
        n_parts x n_parts np.array : VI distance matrix between partitions
    """
    
    num_partitions,n=np.shape(partitions)
    nodes = np.arange(n)
    c=len(np.unique(partitions))
    vi_mat=np.zeros((num_partitions,num_partitions))
    
    print('Calculating VI...')
    
    for i in tqdm(range(num_partitions)):
        # if i%250==0:
        #     print(i)
        A1 = sparse.coo_matrix((np.ones(n),(partitions[i,:],nodes)),shape=(c,n),dtype=np.uint).tocsc()
        n1all = np.array(A1.sum(1),dtype=float)
        
        for j in range(i):
            
            A2 = sparse.coo_matrix((np.ones(n),(nodes,partitions[j,:])),shape=(n,c),dtype=np.uint).tocsc()
            n2all = np.array(A2.sum(0),dtype=float)
            
            n12all = np.array(A1.dot(A2).todense(),dtype=float)
            
            rows, columns = n12all.nonzero()
            
            nmat = np.divide(n12all*n12all,(np.outer(n1all,n2all)),where=np.outer(n1all,n2all)>0)
            nmat[np.outer(n1all,n2all)==0]=1

            vi = np.sum(n12all[rows,columns]*np.log((nmat[rows,columns])))
            
            vi = -1/n*vi
            vi_mat[i,j]=vi
            vi_mat[j,i]=vi
    
    print("Finished!")
    return vi_mat

#Perform embedding into a 2D space using MDS
def embedding(vi_mat,LL,n_neighbors=10,deg_corr=False):
    n_components=2
    Y = manifold.MDS(n_components,dissimilarity='precomputed').fit_transform(vi_mat)
    
    color=np.zeros(1000)
    color[:6]=np.ones(6)
    
    #~ plt.figure()
    #~ plt.plot(Y[:, 0], Y[:, 1], 'k.')
    #~ plt.plot(Y[-n_close:, 0], Y[-n_close:, 1], 'r.')
    #~ for i in xrange(6):
        #~ plt.plot(Y[i, 0], Y[i, 1], 'bo',ms=3+3*i)
    #~ plt.scatter(Y[:, 0], Y[:, 1], c=LL)
    
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], LL[:,0], c=LL[:,0], marker='o')
    fig.show()
    
    if deg_corr:
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], LL[:,1], c=LL[:,1], marker='o')
        fig.show()
    
    return Y


def run_partitions(partitions,edges,nreps=15,divisions=20,outFile=None,DC=False,LL_fun=None):
    """
    Given a base set of partitions, sample many more partitions around these so as to generate 3D plots of partition space.

    Args:
        partitions (n_parts x N np.array): set of partitions, with label for node j in partition i at (i,j) 
        edges (m x 2 np.array): edgelist 
        nreps (int, optional): Number of repetitions of randomisation. Defaults to 15.
        divisions (int, optional): Number of divisions to apply to each real partition (?). Defaults to 20.
        outFile ([type], optional): File path to write output. Defaults to None.
        DC (bool, optional): Degree-corrected? Defaults to False.
        LL_fun (function, optional): Function which takes a partition (timestep passed in as second argument with call) and returns log likelihood term to include. 

    Returns:
        vi_mat, LL, partitions, Y: Get VI distance matrix between full set of partitions, the loglikelihood 
        for each partition, the set of partitions themselves, and the embedding found.
    """
    n_random=500
    if -1 in np.unique(partitions):
        raise ValueError("-1 found in partition: remove non-present nodes before passing to this function")
    c=len(np.unique(partitions))
    n=np.shape(partitions)[1]
    n_partitions=np.shape(partitions)[0]
    partitions=np.append(partitions,np.random.randint(0,c,(n_random,n)),0)
    
    n_close=n_partitions*divisions*nreps
    
    partitions=np.append(partitions,np.zeros((n_close,n)),0)
    
    print(f"Aiming for {np.shape(partitions)[0]} partitions for {np.shape(partitions)[1]} nodes") 
    print(f"Including {n_random} fully random partitions with same number of groups as baseline")
    
    pi=n_partitions+n_random
    reps=range(nreps)
    steps_size=n/divisions
    n_lim=(steps_size)*divisions
    # print(f"Stepsize: {steps_size}, n_lim: {n_lim}")
    #~ if n%20>0:
        #~ n_lim-=steps_size
    
    print(f"Provided {n_partitions} original partitions, each of which will have sections \
          randomised {len(np.arange(0,n_lim,steps_size))} times,\n{n} nodes, {n_lim} n_lim, step size for number of randomised labels is {steps_size}. \
          \nHence in total will compare {n_partitions*len(np.arange(0,n_lim,steps_size))*nreps+n_random+n_partitions} partitions")
    
    for i in range(n_partitions):
        for si in np.arange(0,n_lim,steps_size):
            for rep in reps:
                #randomly select a partition j
                j=sample(range(n_partitions),1)[0]
                #select labels to fix to partition j
                nj=sample(range(n),1)[0]
                inds = sample(range(n),nj)
                #combine partitions i and j
                partitions[pi,:] = partitions[i,:]
                partitions[pi,inds] = partitions[j,inds]
                # print(f"si {si}",f"n {n}")
                inds = sample(range(n),int(si))
                partitions[pi,inds] = np.random.randint(0,c,int(si))
                pi+=1
    
    
    LL=calcLL(partitions,edges,LL_fun=LL_fun)
    print(f"Finished generating {pi} partitions")
    
    order=sample(range(1000),1000)
    
    vi=calcVI(partitions)
    Y=embedding(vi,LL)
    if outFile is not None:
        with open(outFile,'w') as f:
            for yi,zi in zip(Y,LL):
                #~ f.write('%f %f %f %f\n' % (yi[0],yi[1],zi[0],zi[1]))
                f.write('%f %f ' % (yi[0],yi[1]))
                for ll in zi:
                    f.write('%f ' % ll)
                f.write('\n')
    
    return vi,LL,partitions,Y