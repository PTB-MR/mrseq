# Relaxometry

Relaxometry refers to the measurement of MR relaxation times (e.g. T1 or T2). Multiple images with different acquisition parameters are 
acquired and a signal model is then used to estimate the relevant relaxation times. 

The most accurate approaches to estimate relaxation time obtain a single readout followed by a long waiting time (i.e. long repetition time). 
This ensures that the data acquisition of one k-space line does not influence the signal in the following k-space line. 
Available examples are:

- [T1 mapping using an inversion pulse and a single line spoiled gradient echo readout](t1_inv_rec_gre_single_line.ipynb)
- [T1 mapping using an inversion pulse and a single line spin echo readout](t1_inv_rec_se_single_line.ipynb)