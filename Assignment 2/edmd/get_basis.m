function psi = get_basis(X, basis)
% Input
% X         : states
% n         : number of basis
% output
% basis     : basis functions

%% build observables [x,basis]
psi = []; %  basis function [linear, nonlienar]

if(strcmp(basis.type,'monomials'))
    deg = basis.deg; % degeree of monomial
    dim = basis.dim; % state dimension
    psi = monomial_basis(X,deg,dim);

elseif(strcmp(basis.type,'rbf'))
    gamma = basis.gamma;
    
    % unifom sampling
    grid_x= -2:0.2:2;
    [c1,c2] = meshgrid(grid_x);
    centers = [c1(:),c2(:)];
    
    % gaussian rbf
    psi =RBF_guassian_basis(X, centers, gamma);

    % thin plate rbf
%     psi = RBF_thin_plate_basis(X, centers);

end




