function out = CE_VALSE_EP_all(y_q,m,ha,T,pilot,h_orig, yy_min,B,alpha,Iter_max,method_EP)
    
    M = size(m,1);
%     N = m(M)+1;
    N = length(h_orig);
    L = N;
%     ratio = min(B/T,1);
% ratio = 1/T;
%  P_omega_vec = P_omega(:);
%     h_var_A_ext = 1e1*ones(M,1);
%     h_mean_A_ext = zeros(M,1);
%     v_B_ext = 1e1*ones(M*T,1);
    v_B_ext = 1e1*ones(M*T,1);
    z_B_ext = zeros(M*T,1);
%     u_A_post = zeros(size(h));
    mse = zeros(Iter_max,1);
    Kt = zeros(Iter_max,1);
    yI = zeros(N,T);
    if B<inf
        y_pre = yy_min + (y_q+0.5)* alpha;
        y_pre_c = y_pre(1:end/2)+1j*y_pre(end/2+1:end);
        yI(m+1,:) = reshape(y_pre_c,M,T);
    else
        y0=y_q;
        yI(m+1,:) = reshape(y0,M,T);
    end
    
R = zeros(N,N,T);
nu_vec = zeros(T,1);
for g = 1:T
    R(:,:,g)  = yI(:,g)*yI(:,g)';
    sR = zeros(N-1,1);
    for i=2:N
        for k=1:i-1
            sR(i-k) = sR(i-k) + R(i,k,g);
        end
    end
    Rh  = toeplitz([sum(diag(R(:,:,g)));sR])/N; 
    try
        evs = sort(real(eig(Rh)),'ascend');
    catch 
         warning('Assigning a value of NaN.');
    end
    nu_vec(g)  = mean(evs(1:floor(N/4)));
end
nu = mean(nu_vec);
% nu = nu_vec(1);
%     nu = y'*y/M/100;
%     nu = 0.4337;
    if B<inf
        z_B_ext_real = [real(z_B_ext);imag(z_B_ext)];
        v_B_ext_real = [v_B_ext;v_B_ext]/2;
        [z_C_post_real, v_C_post_real] = GaussianMomentsComputation(y_q, z_B_ext_real, v_B_ext_real, yy_min, B, alpha, nu/2);
        v_C_post = v_C_post_real(1:M*T)+v_C_post_real(M*T+1:end);
        z_C_post = z_C_post_real(1:M*T)+1j*z_C_post_real(M*T+1:end);
        v_C_ext = v_C_post.*v_B_ext./(v_B_ext-v_C_post);
        v_C_ext = v_C_ext.*(v_C_ext>0)+10*max(v_C_ext).*(v_C_ext<=0);
        z_C_ext = v_C_ext.*(z_C_post./v_C_post-z_B_ext./v_B_ext);
        
        V_C_ext = reshape(v_C_ext,M,T);
        Z_C_ext = reshape(z_C_ext,M,T);
    else
        v_C_ext = nu*ones(M*T,1);
        z_C_ext = y_q;
        V_C_ext = reshape(v_C_ext,M,T);
        Z_C_ext = reshape(z_C_ext,M,T);
    end
%     if T>1
%         for gg = 2:T
%             V_C_ext(~P_omega(:,gg)) = inf;
%         end
%     end
    v_u_B_ext = 1./sum(bsxfun(@rdivide,abs(pilot.').^2,V_C_ext),2);
    u_B_ext = v_u_B_ext.*sum(bsxfun(@times,pilot',Z_C_ext./V_C_ext),2);
%     v_u_B_ext = 1./mean(bsxfun(@rdivide,abs(pilot.').^2,V_C_ext),2);
%     u_B_ext = v_u_B_ext.*mean(bsxfun(@times,pilot',Z_C_ext./V_C_ext),2);
%         v_u_B_ext = 1./sum(bsxfun(@rdivide,abs(pilot(1).').^2,V_C_ext(:,1)),2);
% %     v_u_B_ext = mean(V_C_ext,2);
%     u_B_ext = v_u_B_ext.*sum(bsxfun(@times,pilot(1)',Z_C_ext(:,1)./V_C_ext(:,1)),2);
%     v_u_B_ext = v_u_B_ext/ratio;
%     zzz = abs(pilot.').^2
    y = u_B_ext;
    sigma = v_u_B_ext;


    y2    = (y./sqrt(sigma))'* (y./sqrt(sigma));
    A     = zeros(L,L);
    J     = zeros(L,L);
    h     = zeros(L,1);
    w     = zeros(L,1);
    C     = zeros(L);
    t = 1;

    % Initialization of the posterior pdfs of the frequencies
    res   = y./sqrt(sigma);
    for l=1:L
        % noncoherent estimation of the pdf
        yI = zeros(N,1);
        yI(m+1) = res;
        R  = yI*yI';
        sR = zeros(N-1,1);
        for i=2:N
            for k=1:i-1
                sR(i-k) = sR(i-k) + R(i,k);
            end
        end
        if l==1 % use the sample autocorrelation to initialize the model parameters
            K   = floor(L/2);
            rho = K/L;
    %         tau = (y2/sum(1./sigma)-sum(sigma)/M)/(rho*L);
            tau = (y2-M)/(rho*L)/sum(1./sigma);
        end
        etaI   = 2*sR/(M+mean(sigma)/tau);
        ind    = find(abs(etaI)>0);
        if ha~=3
            [~,mu,kappa] = Heuristic2(etaI(ind), ind, N);
            A(m+1,l) = exp(1i*m * mu) .* ( besseli(m,kappa,1)/besseli(0,kappa,1) );
        else
            [~,mu] = pntFreqEst(etaI(ind), ind);
            A(m+1,l) = exp(1i*m * mu);
        end
        % compute weight estimates; rank one update
        w_temp = w(1:l-1); C_temp = C(1:l-1,1:l-1);
        J(1:l-1,l) = A(m+1,1:l-1)'*diag(1./sigma)*A(m+1,l); J(l,1:l-1) = J(1:l-1,l)'; J(l,l) = sum(1./sigma);
        J = (J+J')/2;
        h(l) = A(m+1,l)'*diag(1./sigma)*y;
        v = 1/(sum(1./sigma)+1/tau-real(J(1:l-1,l)'*C_temp*J(1:l-1,l)));  % diagonal case
        u = v .* (h(l) - J(1:l-1,l)'*w_temp);

        w(l) = u;
        ctemp = C_temp*J(1:l-1,l);
        w(1:l-1) = w_temp - ctemp*u;
        C(1:l-1,1:l-1) = C_temp + v*(ctemp*ctemp');
        C(1:l-1,l) = -v*ctemp;  C(l,1:l-1) = C(1:l-1,l)'; C(l,l) = v;

        % the residual signal
        res = (y - A(m+1,1:l)*w(1:l))./sqrt(sigma);

        if l==K % save mse and K at initialization
            xro    = A(:,1:l)*w(1:l);
            if B==1
                deb_c = xro'*h_orig/(xro'*xro+eps); 
            else
               deb_c = 1;
            end
            mse(t) = 10*log10(norm(h_orig-deb_c*xro)^2/norm(h_orig)^2);
            Kt(t)  = K;
        end
    end

cont = 1;
while cont
%         ratio = max(0.1^t,1/T);
%         if t==102
%             11
%         end
        t = t + 1;
        % Update the support and weights
        [ K, s, w, C ] = maxZ_diag( J, h, M, sigma, rho, tau, T );
        if K==0
            out = struct('noise_var',nu,'iterations',t,'MSE',mse,'K',Kt);
            sprintf('No signal detected');
%             out = struct('freqs',th,'amps',w(s),'x_estimate',xr,'noise_var',nu,'iterations',t,'MSE',mse,'K',Kt);
            return;
        end
        % Update the noise variance, the variance of prior and the Bernoulli probability
        if K>0
            tau = real( w(s)'*w(s)+trace(C(s,s)) )/K;
            if K<L
                rho = K/L;
            else
                rho = (L-1)/L; % just to avoid the potential issue of log(1-rho) when rho=1
            end
        else
            rho = 1/L; % just to avoid the potential issue of log(rho) when rho=0
        end
        inz = 1:L; inz = inz(s); % indices of the non-zero components
        th = zeros(K,1);
        for i = 1:K
            if K == 1
                r = y;
               eta = 2 * ( (r./sigma) * w(inz)' );
            else
                A_i = A(m+1,inz([1:i-1 i+1:end]));
                r = y - A_i*w(inz([1:i-1 i+1:end]));
               eta = 2*diag(1./sigma) * ( r * w(inz(i))' - A_i * C(inz([1:i-1 i+1:end]),i) );
            end
            if ha == 1
                [A(:,inz(i)), th(i)] = Heuristic1( eta, m, 1000 );
            elseif ha == 2
                [A(:,inz(i)), th(i)] = Heuristic2( eta, m, N );
            elseif ha == 3
                [A(:,inz(i)), th(i)] = pntFreqEst( eta, m );
            end
        end
        
        z_A_post  = A(m+1,s)*w(s);

        v_A_post = real(diag(A(m+1,s)*C(s,s)*A(m+1,s)'));
        TT_mat = (A(m+1,s).*conj(A(m+1,s)));
        add_var1 = w(s)'*w(s)*ones(M,1)-TT_mat*(w(s).*conj(w(s)));
        add_var2 = trace(C(s,s))*ones(M,1)-TT_mat*diag(C(s,s));
        v_A_post = v_A_post+real(add_var1)+real(add_var2);
        
        Z_A_post_x = z_A_post*(pilot.');
        V_A_post_x = v_A_post*(abs(pilot.').^2);
        
        nu = mean(abs(z_C_ext-Z_A_post_x(:)).^2+V_A_post_x(:));
        switch method_EP
            case 'scalar_EP'
                v_A_post = mean(v_A_post)*ones(M,1);
            case 'diag_EP'
%                 v_A_post = v_A_post;
        end
        
        v_u_A_post = v_A_post;
        u_A_post = z_A_post;
        v_u_A_ext = v_u_A_post.*sigma./(sigma-v_u_A_post);
        v_u_A_ext = v_u_A_ext.*(v_u_A_ext>0)+10*max(max(v_u_A_ext),1)*(v_u_A_ext<=0);
        u_A_ext = v_u_A_ext.*(u_A_post./v_u_A_post-y./sigma); 
        
        h_mean_A_ext = u_A_ext;
        h_var_A_ext = v_u_A_ext;
        if T==1
            z_B_ext = h_mean_A_ext*(pilot.');
            v_B_ext = h_var_A_ext*(abs(pilot.').^2);
            
        else
            
%             for g = 1:T
%                 V_B_ext0_inv(:,g) = 1./(h_var_A_ext*(abs(pilot(g)).^2))+sum(1./V_C_ext,2)-1./V_C_ext(:,g);
%                 V_B_ext0(:,g) = 1./V_B_ext0_inv(:,g);
%                 Z_B_ext0(:,g) = V_B_ext0(:,g) .*(h_mean_A_ext./h_var_A_ext*pilot(g)./abs(pilot(g))^2+sum(Z_C_ext./V_C_ext,2)-Z_C_ext(:,g)./V_C_ext(:,g));
%             end

            V_B_ext_diff = bsxfun(@minus,sum(1./V_C_ext,2),1./V_C_ext);
            V_B_ext0_inv = bsxfun(@plus,1./(h_var_A_ext*(abs(pilot.').^2)),V_B_ext_diff);
            V_B_ext0 = 1./V_B_ext0_inv;  % h_mean_A_ext*(pilot.')./(h_var_A_ext*(abs(pilot.').^2))
            temp = (sum(Z_C_ext./V_C_ext,2)-Z_C_ext./V_C_ext);
            temp_p = (pilot./(abs(pilot).^2)).';
            Z_B_ext0 = V_B_ext0.*(bsxfun(@plus,(h_mean_A_ext./h_var_A_ext).*temp_p,temp));
           
%             Z_B_ext0 = h_mean_A_ext*(pilot.');
%             V_B_ext0 = h_var_A_ext*(abs(pilot.').^2);
            z_B_ext = Z_B_ext0(:);
            v_B_ext = V_B_ext0(:);
%             v_B_ext = v_B_ext/ratio;
            
        end
            
        if B<inf
%             t
            z_B_ext_real = [real(z_B_ext);imag(z_B_ext)];
            v_B_ext_real = [v_B_ext;v_B_ext]/2;
            [z_C_post_real, v_C_post_real] = GaussianMomentsComputation(y_q, z_B_ext_real, v_B_ext_real, yy_min, B, alpha, nu/2);
            v_C_post = v_C_post_real(1:M*T)+v_C_post_real(M*T+1:end);
            z_C_post = z_C_post_real(1:M*T)+1j*z_C_post_real(M*T+1:end);
            v_C_ext = v_C_post.*v_B_ext./(v_B_ext-v_C_post);
%             v_C_ext = min(v_C_ext,1e4);
            v_C_ext = v_C_ext.*(v_C_ext>0)+max(v_C_ext).*(v_C_ext<=0);
            z_C_ext = v_C_ext.*(z_C_post./v_C_post-z_B_ext./v_B_ext);
%             nu = mean(abs(z_C_ext-z_C_post).^2+v_C_post);
%             if T==1
%                 nu = mean(abs(z_C_ext-z_C_post).^2+v_C_post);
%             else
%                 nu = mean(abs(z_C_ext(P_omega_vec==1)-z_C_post(P_omega_vec==1)).^2+v_C_post(P_omega_vec==1));
%             end
                
                
%             nu = median(abs(z_C_ext-z_C_post).^2+v_C_post);
        else
            v_C_post =v_B_ext.*nu./(v_B_ext+nu);
            z_C_post = v_C_post.*(z_B_ext./v_B_ext+y_q./nu);
            v_C_ext = nu*ones(M*T,1);
            z_C_ext = y_q;
%             nu = mean(abs(z_C_ext-z_C_post).^2+v_C_post);
        end
        V_C_ext = reshape(v_C_ext,M,T);
%         if T>1
%              for gg = 2:T
%                 V_C_ext(~P_omega(:,gg)) = inf;
%             end
% %             V_C_ext(~P_omega) = inf;
%         end
        Z_C_ext = reshape(z_C_ext,M,T);
        v_u_B_ext = 1./sum(bsxfun(@rdivide,abs(pilot.').^2,V_C_ext),2);
        u_B_ext = v_u_B_ext.*sum(bsxfun(@times,pilot',Z_C_ext./V_C_ext),2);
%         v_u_B_ext = 1./mean(bsxfun(@rdivide,abs(pilot.').^2,V_C_ext),2);
%         u_B_ext = v_u_B_ext.*mean(bsxfun(@times,pilot',Z_C_ext./V_C_ext),2);
%         v_u_B_ext = 1./sum(bsxfun(@rdivide,abs(pilot(1).').^2,V_C_ext(:,1)),2);
%         u_B_ext = v_u_B_ext.*sum(bsxfun(@times,pilot(1)',Z_C_ext(:,1)./V_C_ext(:,1)),2);
%         v_u_B_ext = v_u_B_ext*T;
        sigma = v_u_B_ext;
        y = u_B_ext;
        J = A(m+1,:)'*diag(1./sigma)*A(m+1,:);
        J = J - diag(diag(J)) +  sum(1./sigma)*eye(N);
        h   = A(m+1,:)'*diag(1./sigma)*y;

        xr     = A(:,s)*w(s);
        if B==1
           deb_c = xr'*h_orig/(xr'*xr+eps); 
        else
           deb_c = 1;
        end
        mse(t) = 10*log10(norm(h_orig-deb_c*xr)^2/norm(h_orig)^2);
        Kt(t)  = K;
        
    % stopping criterion:
    % the relative change of the reconstructed signalis below threshold or
    % max number of iterations is reached
        if (norm(xr-xro)/norm(xro)<1e-6) || (norm(xro)==0&&norm(xr-xro)==0) || (t >= Iter_max)||nu>1e60
            cont = 0;
            mse(t+1:end) = mse(t);
            Kt(t+1:end)  = Kt(t);
        end
        xro = xr;
end
out = struct('freqs',th,'amps',w(s),'x_estimate',xr,'noise_var',nu,'iterations',t,'MSE',mse,'K',Kt);
end    
function [a, theta, kappa, mu] = Heuristic1( eta, m, D )
%Heuristic1 Uses the mixture of von Mises approximation of frequency pdfs
%and Heuristic #1 to output a mixture of max D von Mises pdfs

M     = length(m);
tmp   = abs(eta);
A     = besseli(1,tmp,1)./besseli(0,tmp,1);
kmix  = Ainv( A.^(1./m.^2) );
[~,l] = sort(kmix,'descend');
eta_q = 0;
for k=1:M
    if m(l(k)) ~= 0
        if m(l(k)) > 1
            mu2   = ( angle(eta(l(k))) + 2*pi*(1:m(l(k))).' )/m(l(k));
            eta_f = kmix(l(k)) * exp( 1i*mu2 );
        else
            eta_f = eta(l(k));
        end
        eta_q = bsxfun(@plus,eta_q,eta_f.');
        eta_q = eta_q(:);
        kappa = abs(eta_q);
        
        % to speed up, use the following 4 lines to throw away components
        % that are very small compared to the dominant one
        kmax  = max(kappa);
        ind   = (kappa > (kmax - 30) ); % corresponds to keeping those components with amplitudes divided by the highest amplitude is larger than exp(-30) ~ 1e-13
        eta_q = eta_q(ind);
        kappa = kappa(ind);
        
        if length(eta_q) > D
            [~, in] = sort(kappa,'descend');
            eta_q   = eta_q(in(1:D));
        end
    end
end
kappa   = abs(eta_q);
mu      = angle(eta_q);
kmax    = max(kappa);
I0reg   = besseli(0,kappa,1) .* exp(kappa-kmax);
Zreg    = sum(I0reg);
n       = 0:1:m(end);
[n1,k1] = meshgrid(n, kappa);
a       = sum( (diag(exp(kappa-kmax))* besseli(n1,k1,1) /Zreg ).*exp(1i*mu*n),1).';
theta   = angle(sum( (diag(exp(kappa-kmax))* besseli(1,kappa,1) /Zreg ).*exp(1i*mu*1),1));
end

function [a, theta, kappa] = Heuristic2( eta, m, N_all )
%Heuristic2 Uses the mixture of von Mises approximation of frequency pdfs
%and Heuristic #2 to output one von Mises pdf

N     = length(m);
ka    = abs(eta);
A     = besseli(1,ka,1)./besseli(0,ka,1);
kmix  = Ainv( A.^(1./m.^2) );
k     = N;
eta_q = kmix(k) * exp( 1i * ( angle(eta(k)) + 2*pi*(1:m(k)).' )/m(k) );
for k = N-1:-1:1
    if m(k) ~= 0
        phi   = angle(eta(k));
        eta_q = eta_q + kmix(k) * exp( 1i*( phi + 2*pi*round( (m(k)*angle(eta_q) - phi)/2/pi ) )/m(k) );
    end
end
[~,in] = max(abs(eta_q));
mu     = angle(eta_q(in));
d1     = -imag( eta' * ( m    .* exp(1i*m*mu) ) );
d2     = -real( eta' * ( m.^2 .* exp(1i*m*mu) ) );
if d2<0 % if the function is locally concave (usually the case)
    theta  = mu - d1/d2;
    kappa  = Ainv( exp(0.5/d2) );
else    % if the function is not locally concave (not sure if ever the case)
    theta  = mu;
    kappa  = abs(eta_q(in));
end
% n      = (0:1:m(end))';
n = (0:1:N_all-1)';
a      = exp(1i*n * theta).*( besseli(n,kappa,1)/besseli(0,kappa,1) );
end

function [a, theta] = pntFreqEst( eta, m )
%pntFreqEst - point estimation of the frequency

th     = -pi:2*pi/(100*max(m)):pi;

[~,i]  = max(real( eta'*exp(1i*m*th) ));
mu     = th(i);
d1     = -imag( eta' * ( m    .* exp(1i*m*mu) ) );
d2     = -real( eta' * ( m.^2 .* exp(1i*m*mu) ) );
if d2<0 % if the function is locally concave (usually the case)
    theta  = mu - d1/d2;
else    % if the function is not locally concave (not sure if ever the case)
    theta  = mu;
end
a      = exp(1i*(0:1:m(end))' * theta);
end

function [ K, s, w, C ] = maxZ_diag( J, h, M, sigma, rho, tau, T )
%maxZ maximizes the function Z of the binary vector s, see Appendix A of
%the paper

L = size(h,1);
% cnst = log(rho/(1-rho))+T*log(1/tau);
cnst = log(rho/(1-rho))+log(1/tau);

K = 0; % number of components
s = false(L,1); % Initialize s
w = zeros(L,1);
C = zeros(L);
u = zeros(L,1);
v = zeros(L,1);
Delta = zeros(L,1);
if L > 1
    cont = 1;
    while cont
        if K<M-1
            v(~s) = 1 ./ ( sum(1./sigma) + 1/tau - real(sum(J(s,~s).*conj(C(s,s)*J(s,~s)),1)) ); % diagonal
            u(~s) = v(~s) .* ( h(~s) - J(s,~s)'*w(s));
%             Delta(~s) = log(v(~s)) + u(~s).*conj(u(~s))./v(~s) + cnst/T;
            Delta(~s) = log(v(~s)) + u(~s).*conj(u(~s))./v(~s) + cnst;
        else
            Delta(~s) = -1; % dummy negative assignment to avoid any activation
        end
        if ~isempty(h(s))
%             Delta(s) = -log(diag(C(s,s))) - w(s).*conj(w(s))./diag(C(s,s)) - cnst/T;
            Delta(s) = -log(diag(C(s,s))) - w(s).*conj(w(s))./diag(C(s,s)) - cnst;
        end
        [~, k] = max(Delta);
        if Delta(k)>0
            if s(k)==0 % activate
                w(k) = u(k);
                ctemp = C(s,s)*J(s,k);
                w(s) = w(s) - ctemp*u(k);
                C(s,s) = C(s,s) + v(k)*(ctemp*ctemp');
                C(s,k) = -v(k)*ctemp;
                C(k,s) = C(s,k)';
                C(k,k) = v(k);
                s(k) = ~s(k); K = K+1;
            else % deactivate
                s(k) = ~s(k); K = K-1;
                w(s) = w(s) - C(s,k)*w(k)/C(k,k);
                C(s,s) = C(s,s) - C(s,k)*C(k,s)/C(k,k);
            end
            C = (C+C')/2; % ensure the diagonal is real
        else
            break
        end
    end
elseif L == 1
    if s == 0
        v = 1 ./ ( sum(1./sigma) + 1/tau );
        u = v * h;
        Delta = log(v) + u*conj(u)/v + cnst;
        if Delta>0
            w = u; C = v; s = 1; K = 1;
        end
    else
        Delta = -log(C) - w*conj(w)/C - cnst;
        if Delta>0
            w = 0; C = 0; s = 0; K = 0;
        end
    end
end
end

function [ k ] = Ainv( R )
% Returns the approximate solution of the equation R = A(k),
% where A(k) = I_1(k)/I_0(k) is the ration of modified Bessel functions of
% the first kind of first and zero order
% Uses the approximation from
%       Mardia & Jupp - Directional Statistics, Wiley 2000, pp. 85-86.
%
% When input R is a vector, the output is a vector containing the
% corresponding entries

k   = R; % define A with same dimensions
in1 = (R<.53); % indices of the entries < .53
in3 = (R>=.85);% indices of the entries >= .85
in2 = logical(1-in1-in3); % indices of the entries >=.53 and <.85
R1  = R(in1); % entries < .53
R2  = R(in2); % entries >=.53 and <.85
R3  = R(in3); % entries >= .85

% compute for the entries which are < .53
if ~isempty(R1)
    t      = R1.*R1;
    k(in1) = R1 .* ( 2 + t + 5/6*t.*t );
end
% compute for the entries which are >=.53 and <.85
if ~isempty(R2)
    k(in2) = -.4 + 1.39*R2 + 0.43./(1-R2);
end
% compute for the entries which are >= .85
if ~isempty(R3)
    k(in3) = 1./( R3.*(R3-1).*(R3-3) );
end

end