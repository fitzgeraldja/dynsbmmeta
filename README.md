# dynsbmmeta
R package defining a dynamic stochastic block model for temporal networks with nodal metadata.

This builds on the CRAN package dynsbm as a base, significantly increasing functionality both for conventional dynamic networks by allowing
- Poisson weighted edges;
- Degree-correction;
- Intermediary saves;
- Return of estimated node marginals;
- Link prediction;

and importantly permitting the inclusion of mixed nodal metadata, through defining one of the following distributions over each type:
- Multivariate Normal;
- Poisson;
- Categorical;
- Independent Bernoulli;
- Negative binomial (in progress).
