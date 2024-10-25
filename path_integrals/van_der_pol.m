%% Eigenfunctions for Van der Pol System
% This script computes and visualizes eigenfunctions for the linearized 
% Van der Pol system using a path integral method.

clc; clear; close all;
set(0,'DefaultLineLineWidth',2)  % Set default line width for plots
set(0,'defaultfigurecolor',[1 1 1])  % Set figure background color to white

%% System Description
% Define the nonlinear system: x_dot = f(x)
Dom = [-3 3];  % Domain for grid
x = sym('x',[2;1],'real');  % Define state variables x1 and x2
mu = 1;  % Van der Pol parameter
alpha = 1;  % Scaling factor
f = alpha*[x(2); mu*x(2) - x(1) - mu*x(1)^2*x(2)];  % Nonlinear dynamics

%% Linearization at (0,0)
% Linearize the system at the equilibrium point (0,0)
eqb_point = [0 0];  % Equilibrium point
A = eval(subs(jacobian(f),[x(1) x(2)],eqb_point));  % Jacobian of f at (0,0)

% Eigen-decomposition of A
[~,Dc,Wc] = eig(A);  % Dc: Eigenvalues, Wc: Eigenvectors
lc1 = Dc(1,1);  % First eigenvalue
wc1 = Wc(:,1);  % First eigenvector

% Convert to real Jordan form
[Wr,Dr] = cdf2rdf(Wc,Dc);  % Wr: Real eigenvectors, Dr: Real eigenvalues

% Define real and imaginary parts of eigenvalues and eigenvectors
evc1_r = real(Dc(1)); evc1_i = imag(Dc(1));
wc1_r = real(Wc(:,1)); wc1_i = imag(Wc(:,1));

% Define the nonlinear part of the dynamics: x_dot = Ax + fn(x)
fn = f - A*x;

% Create MATLAB functions for computations
wc1Fn = matlabFunction(wc1'*fn,'vars',{x(1),x(2)});
WrFn = matlabFunction(Wr'*fn,'vars',{x(1),x(2)});
f_fun = matlabFunction(f);

%% Path Integral Setup
dim = 1;  % Dimension for integration (1 for scalar)

% Define grid where eigenfunction is well-defined
bounds = Dom(2);  % Upper bound for grid
grid_x = -bounds:0.1:bounds;  % Create grid
[xx1,xx2] = meshgrid(grid_x);  % Generate meshgrid

% Initialize matrices for eigenfunction computations
phi1_dir = zeros(size(xx2)); 
phi2_matlab = zeros(size(xx2));
Phi_rf = zeros([size(xx2),2]);  % For realified eigenfunctions

% ODE solver options (terminate if the solution goes off-grid)
options = odeset('RelTol',1e-9,'AbsTol',1e-9, ...
    'events',@(t, x)offFrame(t, x, Dom(2)));
tEnd = 30;  % Integration time horizon
tspan = [0 -sign(real(Dc(1,1)))*tEnd];  % Time span for backward integration

% Loop over all grid points
w_bar = waitbar(0,'1','Name','Calcualting path integral...',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');

for iX1 = 1:size(xx1,1)
    waitbar(iX1/length(xx1),w_bar,sprintf(string(iX1)+'/'+string(length(xx1))))
    for iX2 = 1:size(xx2,2)
        x_i = [xx1(iX1,iX2); xx2(iX1,iX2)];  % Current grid point
        
        % Solve system dynamics using ODE45
        [t,y] = ode45(@(t,y)f_fun(y(1),y(2)),tspan,x_i,options);
        
        % Check if solution is valid (near equilibrium)
        if norm(y(end,:)) > 2e-6
            phi1_dir(iX1,iX2) = 0;
            Phi_rf(iX1,iX2,:) = [0;0];
        else
            % Compute eigenfunction using path integral method
            phi1_dir(iX1,iX2) = wc1'*x_i + ...
                trapz(t,exp(lc1*t).*wc1Fn(y(:,1),y(:,2)), dim);

            % Compute realified eigenfunctions
            Phi_rf(iX1,iX2,:) = Wr'*x_i + getIntegral(t,y,Dr,WrFn);
        end
    end
end
F = findall(0,'type','figure','tag','TMWWaitbar');
delete(F);

%% Plot Eigenfunctions
% Set up for plotting
close all;

% Absolute value of realified eigenfunctions
figure(1)
subplot(1,2,1)
phi_r2c = Phi_rf(:,:,1) + 1i*(Phi_rf(:,:,2));  % Combine real parts
surf_vals1 = abs(phi_r2c);  % Absolute value
surf_vals1(surf_vals1 > 25) = 25;  % Cap large values for clarity

% Plot contours for eigenfunction magnitudes
levels1 = logspace(-3,1.4,20);
s1 = contourf(xx1,xx2,surf_vals1,levels1); hold on
axis square; set(gca,'FontSize',15);
xlabel('$x_1$','FontSize',20, 'Interpreter','latex')
ylabel('$x_2$','FontSize',20, 'Interpreter','latex')
zlabel('$|\phi(x)|$','Interpreter','latex','FontSize',20)
box on; grid on;
set(gca,'LineWidth',2); colorbar
xlim([-2.6,2.6]); ylim([-2.6,2.6]);
view(2)
title('Eigenfunction Magnitude')

% Phase of realified eigenfunctions
subplot(1,2,2)
surf_vals2 = angle(phi_r2c);  % Phase angle
surf_vals2(surf_vals2==0)=-15;
levels2 = linspace(-2*pi,2*pi,25);
s2 = contourf(xx1,xx2,surf_vals2,levels2); hold on
colormap jet

axis square; set(gca,'FontSize',15);
xlabel('$x_1$','FontSize',20, 'Interpreter','latex')
ylabel('$x_2$','FontSize',20, 'Interpreter','latex')
zlabel('$\angle\phi(x)$','Interpreter','latex','FontSize',20)
box on; grid on;
set(gca,'LineWidth',2); colorbar
xlim([-2.6,2.6]); ylim([-2.6,2.6]);
view(2)
title('Eigenfunction Phase')

%% Helper Functions

% Event function for ODE solver: stop if solution goes off-frame
function [value,isterminal,direction] = offFrame(~, Y, Dom)
    value = (max(abs(Y)) > 10*Dom) | (norm(Y) < 1e-6);  % Off-frame condition
    isterminal = 1;  % Stop integration
    direction = 0;
end

% Compute path integral for realified eigenfunctions
function integralVal = getIntegral(t, y, D, wFn)
    iTraj = zeros(2,length(t));
    for iT = 1:length(t)
        iTraj(:,iT) = expm(abs(t(iT))*D)*wFn(y(iT,1),y(iT,2));
    end
    integralVal = trapz(t,iTraj,2);  % Numerical integration
end
