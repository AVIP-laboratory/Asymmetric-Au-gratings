# Asymmetric-Au-gratings

This project focuses on learning-based wave propagation approaches for optimizing asymmetric double-layer gold (Au) grating structures. The goal is to identify geometric parameter combinations that maximize TM transmission and enhance polarization performance.

## Datasets description

you can find data folder

**Datasets**
geometric_parameters_train.csv 파일은 p_u and p_l and t_d are varied from 0.05 to 1 µm in steps of 0.05 µm, resulting in a total of 2600개 조합
simulated_TM_results.csv 파일은 The transmission spectra of 2-6 µm with 0.1 µm increments for all parameter combinations 
geometric_parameters_test.csv 파일은  p_u and p_l and t_d are varied from 0.05 to 1 µm in steps of 0.01 µm, resulting in a total of 884736개 조합 이건 예시로 0.01 step으로 한거고 사용자 정의에 맞게 더 세밀한 조합에 대해서 결과 예측 가능함.

the geometric parameters of the structure are defined as follows: upper grating period (pu), lower grating period (pl) and dielectric spacer thickness (t_d). 

![그림1](/data/그림1.png)


