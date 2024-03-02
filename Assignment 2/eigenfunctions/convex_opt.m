function [phi1, phi2] = convex_opt(f,A0,basis,X_samples)
% function to approximate principle eigenfunctions directly from data
    Psi_fun = basis.fn;
    DPsi_fun = basis.fn_grad;
    Nbs = basis.Nbs;

    [~, D_c, W_c] = eig(A0);
    [W_r, D_r] = cdf2rdf(W_c, D_c);
    [~,ind] = sort(real(diag(D_r)));
    D_r = D_r(ind,ind);
    W_r = W_r(:,ind);
    Lam1 = D_r(1,1); % D(2,2); % corresponding to the linear EF
    Lam2 = D_r(2,2); % D(1,1); % corresponding to the nonlinear EF
    W1_ = -W_r(:,1)/min(abs(W_r(:,1)));
    W2_ = W_r(:,2)/min(abs(W_r(:,2)));
    Fx = @(x)f(0, x);
    Fnx = @(x)f(0, x)-A0*x;
    % 1st EF
    J1_mu_bar = zeros(Nbs);
    b1_mu_bar = zeros(size(J1_mu_bar,1),1);
    % 2nd EF
    J2_mu_bar = zeros(Nbs);
    b2_mu_bar = zeros(size(J2_mu_bar,1),1);
    for pt_i=1:size(X_samples,2)
        Gam_M_i = Psi_fun(X_samples(:,pt_i));
        dGam_M_dx_i = DPsi_fun(X_samples(:,pt_i));
        F_x_i = Fx(X_samples(:,pt_i));
        Fn_x_i = Fnx(X_samples(:,pt_i));
        % 1st EF
        J1_mu_bar_i = Gam_M_i*(dGam_M_dx_i*F_x_i - Lam1*Gam_M_i)';
        J1_mu_bar = J1_mu_bar + J1_mu_bar_i;
        b1_mu_bar_i = W1_'*Fn_x_i*Gam_M_i;
        b1_mu_bar = b1_mu_bar + b1_mu_bar_i;
        % 2nd EF
        J2_mu_bar_i = Gam_M_i*(dGam_M_dx_i*F_x_i - Lam2*Gam_M_i)';
        J2_mu_bar = J2_mu_bar + J2_mu_bar_i;
        b2_mu_bar_i = W2_'*Fn_x_i*Gam_M_i;
        b2_mu_bar = b2_mu_bar + b2_mu_bar_i;
    end
    
    % 1st EF
    J1_mu_bar = J1_mu_bar/(size(X_samples,2));
    b1_mu_bar = -b1_mu_bar/(size(X_samples,2));
    % Galerkin projection coefficients (u* from paper)
    hat_h1 = J1_mu_bar\b1_mu_bar;
    % 2nd EF
    J2_mu_bar = J2_mu_bar/(size(X_samples,2));
    b2_mu_bar = -b2_mu_bar/(size(X_samples,2));
    % Galerkin projection coefficients (u* from paper)
    hat_h2 = J2_mu_bar\b2_mu_bar;
    
    % 1st EF
    phi1 = @(x) W1_'*x +  (hat_h1'*Psi_fun(x));
    % 2nd EF
    phi2 = @(x) W2_'*x +  (hat_h2'*Psi_fun(x));
end