function Psi = RBF_gaussian_basis(x, centers, gamma)
%  [Psi, DPsi] = GaussianRBF_basis(deg, dim) returns a gaussian RBF function and its derivative 
%   There is not a 1 included in Psi, only the linear and higher order terms
%   Centers = matrix of center points (size = (dim of x, K))
%   dim = number of states

% x=sym('x',[dim,1]);
% assume(x,'real')

Psi = [];
G = [];

for i=1:size(centers,1)
    r_squared  = norm(x-centers(i,:))^2;
    g = exp(-gamma^2 * r_squared);
    G = [G; g];
end

Psi = [x;G];

end