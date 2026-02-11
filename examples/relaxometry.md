# Relaxometry

Relaxometry refers to the measurement of MR relaxation times (e.g. T1 or T2). Multiple images with different acquisition parameters are acquired and afterwards a signal model is used to estimate the relevant quantitative relaxation times. 

The most accurate approaches to estimate relaxation times obtain a single readout followed by a long waiting time (i.e. long repetition time). This ensures that the data acquisition of one k-space line does not influence the signal in the following k-space line.

The following examples show how to create the sequences, simulate the data acquisition with [MRzero](https://github.com/MRsources/MRzero-Core), reconstruct the qualitative images and estimate the quantitative relaxation times using [MRpro](https://github.com/PTB-MR/MRpro).

Available examples are:

- [T1 mapping using an inversion pulse and a single line spoiled gradient echo readout](t1_inv_rec_gre_single_line.ipynb)
- [T1 mapping using an inversion pulse and a single line spin echo readout](t1_inv_rec_se_single_line.ipynb)
- [T1 mapping using Modified Look-Locker Inversion recovery (MOLLI)](t1_molli_bssfp.ipynb)
- [T1 and T2 mapping using a spiral cardiac MR Fingerprinting sequence](t1_t2_spiral_cmrf.ipynb)
- [T2 mapping using a multi-echo spin echo readout](t2_multi_echo_se_single_line.ipynb)
- [T2 mapping using a FLASH sequence with T2-preparation pulses](t2_t2prep_flash.ipynb)
- [T2* mapping using a multi-echo FLASH sequence](t2star_multi_echo_flash.ipynb)
- [T1rho mapping using a spin-lock preparation pulse and a single line spin echo readout](t1rho_se_single_line.ipynb)

