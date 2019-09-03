# GL-QVBCE
Perform gridless channel estimation using angular structures
This code is for the following papers:
Jiang Zhu, Chao-kai Wen, Jun Tong, Chongbin Xu and Shi Jin, 
Gridless Variational Bayesian Channel Estimation for Antenna Array Systems with Low Resolution ADCs,
https://arxiv.org/abs/1906.00576

This code is built based on the VALSE algorithm, and is written by Jiang Zhu. If you have any problems about the code, please feel free to contact jiangzhu16@zju.edu.cn


Please directly run NMSEvsIter to see the performance of the GL_QVBCE. 


Main function：

Out=CE_VALSE_EP_all(y_q,m,ha,T,pilot,h_orig,yy_min,B,alpha,Iter_max,method_EP)

Input parameters：
y_q：For quantization, it belongs to $0,1,2,\cdots,2^B-1$. For unquantized setting, it is the unquantized measurmenets.
m: The index correspond to incomplete measurements. 
ha: Set ha=2, which corresponds to Heuristic 2
T: The number of pilots
h_orig: True channel (spectral signal)
Iter_max：The maximum number of iterations
B: The bit-depth. For unquantized setting, set B=inf.
yy_min: The left endpoints of the quantizer
alpha: The stepsize of the quantizer
method_EP：'diag_EP' or scalar_EP’. Please set 'diag_EP'.

Output parameters
out=struct('freqs',th,'amps',w(s),'x_estimate',xr,'noise_var',nu,'iterations',t,'MSE',mse,'K',Kt);
freqs: The point estimates of the frequency
amps: The complex weight amplitude
xr：Reconstructed channel (line spectral)
noise_var: noise variance estimate
t: The iterations exited 
MSE：Normalized MSE
K: Tracking the number of paths during iteration
