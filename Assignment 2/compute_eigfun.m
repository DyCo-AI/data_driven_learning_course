clear; close all; clc;

set(0,'DefaultLineLineWidth',2)
set(0,'defaultfigurecolor',[1 1 1])

addpath dynamics edmd eigenfunctions

%% Problem setup
% Dynamics
system.n = 2;
x = sym('x',[system.n;1]); 
t = sym('t',1);

%%%%%%%%%%%%%%%%% Set up true eigenfunctions %%%%%%%%%%%%%%%%%%%%%
% complete code block to obtain true eig vals and eigfns
% true eigval (diagonal matrix)
%lambda = ??

% true eigenfunctions (inline function)
%Phi = @(x) [??; ??];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% corresponding system dynamics
dynamics_fn = dynamics_custom(lambda, Phi);
f = matlabFunction(dynamics_fn,'Vars',{t,x});

%%%%%%%%%%%%% Visualize the phase portrait %%%%%%%%%%%%%%%%%%%%
% complete code to plot phase portrait of the system


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Linearize the system

%%%%%%%%%%%%% obtain linear dynamics around eqb point %%%%%%%%%%%
% complete code to obtain linearization of the system

% A0 = ??

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Basis function setup

% define basis setup
    basis.dim = 2; % system dimension

    %%%%%%%%%%%%% Setup basis properties for EDMD %%%%%%%%%%%%
    % complete code
    % specify basis type as a string below ('monomials or rbf')
    % basis.type = ?? 

    switch basis.type
    case 'monomials'
        % specifiy degree of monomial sbelow
        % basis.deg = ??;
        basis.remove_linear = true;
    case 'rbf'
        % specifiy kernel width of rbfs
        % basis.gamma = 0.5;
    otherwise
        disp('Please specify type of basis')
    end

% obtain nonlinear part of basis functions and its jacobian
psi = get_basis(x, basis);
psi_fn = matlabFunction(psi,'Vars',{x});

psi_grad = jacobian(psi,x);
psi_grad_fn = matlabFunction(psi_grad,'Vars',{x});

basis.fn = psi_fn;
basis.fn_grad = psi_grad_fn;
basis.Nbs = length(psi);

%% Generate training data

%%%%%%%%%%%%% sample points in a uniform grid %%%%%%%%%%%%%%%%%%%%
% complete code to generate grid points
% X_samples = ??


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Convex optimization of approximate principl eigfns of system
% use the provided convex_opt fun to obtain approximate eigfuns
[phi1_hat, phi2_hat] = convex_opt(f,A0,basis,X_samples);

%% Compare eigenfunction approximation with true eigenfunctions

%%%%%%%%%%%%%%%%%%%%% Plot true eigenfunctions %%%%%%%%%%%%%%%%%%%
% complete code to generate surface plot of the true eigfns


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% Plot true eigenfunctions %%%%%%%%%%%%%%%%%%%%
% complete code to generate surface plot of the approximated eigfns


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% find avg approximation error %%%%%%%%%%%%%%%%
% complete code to find the avg pointwise error


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
