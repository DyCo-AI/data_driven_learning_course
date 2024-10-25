clc; clear; close all;

%set default figure properties
set(0,'DefaultLineLineWidth',2)
set(0,'defaultfigurecolor',[1 1 1])

% fix random seed
seed = 1;
rng(seed);

% import functions
addpath dynamics dmd edmd training prediction

%% Problem setup
% for duffing, states = [x1, x2].
dynamics_fn = @dynamics_duffing; % point to pendulum dynamics

% operator approximation method
% set to EDMD_flag = false for DMD.
% set to EDMD_flag = true for EDMD.
EDMD_flag = false; 


%% Load training trajectory data using full domain
X = load("training\duffing_multi_model_training.csv");

%%%%%%%%%%%%% Visualize the training dataset %%%%%%%%%%%%%%%%%%%%
% complete code to plot phase portraits of training data


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%% Generate time-shfited snapshots %%%%%%%%%%%%%%%%%%%%
% complete code to obtain snapshots X1 and X2 from dataset X


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% get Koopman operator
if(EDMD_flag)
    % get Koopman operator using EDMD
    % define basis setup
    basis.dim = 2; % system dimension

    %%%%%%%%%%%%% Setup basis properties for EDMD %%%%%%%%%%%%
    % complete code
    % specify basis type as a string below ('monomials or rbf'
    % basis.type = ??  

    switch basis.type
    case 'monomials'
        % specifiy degree of monomial sbelow
        % basis.deg = ??;
    case 'rbf'
        % specifiy kernel width of rbfs
        % basis.gamma = 0.5;
    otherwise
        disp('Please specify type of basis')
    end

    %%%%%%%%%% Obtain Koopman operator using EDMD %%%%%%%%%%%%
        
    % complete code for function get_EDMD()
    [A, C] = get_EDMD(X1, X2, basis);   
    operator.A = A;
    operator.C = C;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

else
    %%%%%%%%%% Obtain Koopman operator using DMD %%%%%%%%%%%%
    basis = [];

    % complete code for function get_DMD()
    A = get_DMD(X1, X2);
    operator.A = A;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%% Plot eigenvalues of Koopman operator
%%%%%%%%% Plot eigenvlaues of A in a unit circle %%%%%%%%%%%%%%%%
% complete code to plot discrete time eigenvalues of A


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% evaluate operator for n timesteps prediction
% Use the following parameters for validation
prediction.n_traj = 9; %traj for evaluations
prediction.n_steps = 20; % num timesteps to predict
prediction.dt = 0.1;
prediction.show_plot = true;

% load the validation dataset
X_eval = load('prediction\duffing_validation.csv');


%%%%%%%%%%%%% Get predictions using Koopman  %%%%%%%%%%%%%%%%%%%%
% complete code for function eval_prediction()
X_pred = eval_prediction(X_eval,operaotr,basis,prediction);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% evaluate the avg rmse and % error across for prediction
X_true = X_eval(1:prediction.n_steps,:);
RMSE = rmse(X_pred, X_true, prediction);
