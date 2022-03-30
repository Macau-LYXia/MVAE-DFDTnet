clc
clear
Kstep = 3;
alpha = 0.98;
str = 'Sim_proteinDisease';
data=load(strcat('../data/',str,'.txt'));
[m,n]=size(data);
Mk = RandSurf(data, Kstep, alpha);
PPMI = GetPPMIMatrix(Mk);
rep_sim1_drug = PPMI;
save(strcat('../code/feature/',str,'.mat'),'rep_sim1_drug'); 