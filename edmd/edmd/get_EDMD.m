function EDMD = get_EDMD(X1, X2, basis)
% function to generate A B matrices using EDMD
% Inputs
% X1            : state trajecotries of n traj of length 1:m-1
% X2            : state trajecotries of n traj of length 2:m
% basis         : structure with properites of basis functions
% Outputs
% A             : n x N lifted Koopman operator
% C             : matrix to map lifted states z to original states x