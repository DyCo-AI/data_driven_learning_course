%% eigenfunctions for 2D analytical system with saddle-point origin
clc; clear; close all;
set(0,'DefaultLineLineWidth',2) %linewidh on plots
set(0,'defaultfigurecolor',[1 1 1])

%% System Description
% Linearization at (0,0) saddle point
domain = [-3 3];
x = sym('x',[2;1]); 
delta = 0.5; 
scaling = 1;

f_x = [x(2); x(1) - delta*x(2) - scaling*x(1)^3]; 
f = matlabFunction(f_x,'vars',{x(1), x(2)});

% Quiver grid setup
[X, Y] = meshgrid(domain(1):0.25:domain(2), domain(1):0.25:domain(2));
u = Y;
v = X - delta*Y - X.^3;

%% Equilibrium Point, Linearization, and Non-linear Part
xEq = [0; 0];
A = eval(subs(jacobian(f_x),[x(1) x(2)]',xEq));
[~, D, W] = eig(A);
[dVal, dIdx] = sort(diag(D), 'descend');

% Arrange D and W in {unstable|stable} order
D = diag([D(dIdx(1), dIdx(1)), D(dIdx(2), dIdx(2))]);
W = [W(:,dIdx(1)), W(:,dIdx(2))];

% Unstable and stable parts
evUnstable = D(1,1);
wUnstable = W(:,1);
evStable = D(2,2);
wStable = W(:,2);

% Non-linear part: x_dot = Ax + fn(x)
fn = f_x - A*x;

% Define MATLAB functions for non-linear terms
wUnstableFn = matlabFunction(wUnstable' * fn, 'vars', {x(1), x(2)});
wStableFn = matlabFunction(wStable' * fn, 'vars', {x(1), x(2)});

%% Path Integral Setup
dim = 1; % Dimension for integration (1 for scalar)

% Define grid where eigenfunction is well defined
grid_x = domain(1):0.1:domain(2);
[xx1, xx2] = meshgrid(grid_x);
ballRadius = 15;

% Initialize matrices for eigenfunctions
phiUnstable_PI = nan(size(xx1));
phiStable_PI = nan(size(xx1));

% ODE solver options
options = odeset('RelTol',1e-8, 'AbsTol',1e-8, 'events', @(t, x)offFrame(t, x, ballRadius));

% Store final time and non-linear part of the eigenfunctions
tf_Unstable = nan(size(phiUnstable_PI));
tf_Stable = nan(size(phiStable_PI));
nl_phiUnstable = nan(size(phiUnstable_PI));
nl_phiStable = nan(size(phiStable_PI));

% Numerical integration to compute eigenfunctions
w_bar = waitbar(0,'1','Name','Calcualting path integral...',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');

for iX1 = 1:size(xx1,1)
    waitbar(iX1/length(xx1),w_bar,sprintf(string(iX1)+'/'+string(length(xx1))))
    for iX2 = 1:size(xx2,2)
        x_i = [xx1(iX1, iX2); xx2(iX1, iX2)];
        
        % Forward time simulation for unstable eigenfunction
        tspan = [0 5];
        [tf, yf] = ode45(@(t, y) f(y(1), y(2)), tspan, x_i, options);
        
        if norm(yf(end,:)) <= ballRadius + 1e-6
            tf_Unstable(iX1,iX2) = tf(end);
            nl_phiUnstable(iX1,iX2) = getIntegral(tf, yf, evUnstable, wUnstableFn);
            phiUnstable_PI(iX1,iX2) = wUnstable' * x_i + nl_phiUnstable(iX1,iX2);
        end
        
        % Backward time simulation for stable eigenfunction
        tspan = [0 25];
        [tr, yr] = ode45(@(t, y) f(y(1), y(2)), -tspan, x_i, options);
        
        if norm(yr(end,:)) <= ballRadius + 1e-6
            tf_Stable(iX1,iX2) = tf(end);
            nl_phiStable(iX1,iX2) = getIntegral(tr, yr, evStable, wStableFn);
            phiStable_PI(iX1,iX2) = wStable' * x_i + nl_phiStable(iX1,iX2);
        end
    end
end
F = findall(0,'type','figure','tag','TMWWaitbar');
delete(F);

%% Check for Eigenfunction Conditions (Unstable Eigenfunction)
e_mLamt_Unstable = exp(-evUnstable * tf_Unstable);
e_h = e_mLamt_Unstable .* nl_phiStable;

%% plot eigenfunctions
close all;
% Unstable eigenfunction
figure(1)
subplot(1,2,1)
surf_vals1 = phiUnstable_PI;
s1 = pcolor(xx1,xx2,surf_vals1);hold on
set(s1,'Edgecolor','none')
colorbar
set(gca,'ColorScale','log')
%plot contour
ic_pts = 1;
l = streamslice(X,Y,u,v); hold on;
set(l,'LineWidth',1)
set(l,'Color','k');
f = @(t,x)[x(2); -delta*x(2) - x(1)^3 + x(1)]; 
bounds = domain(2);
xl = -bounds; xh = bounds;
yl = -bounds; yh = bounds;
for x0 = linspace(-bounds, bounds, ic_pts)
    for y0 = linspace(-bounds, bounds, ic_pts)
        [ts,xs] = ode45(@(t,x)f(t,x),tspan,[x0 y0]);
        plot(xs(:,1),xs(:,2),'k','LineWidth',1); hold on;
    end
end

xlim([-3,3])
ylim([-3,3])
axes = gca;
axis square
set(axes,'FontSize',15);
xlabel('$x_1$','FontSize',20, 'Interpreter','latex')
ylabel('$x_2$','FontSize',20, 'Interpreter','latex')
zlabel('$\phi(x)$','FontSize',20, 'Interpreter','latex')
box on
axes.LineWidth=2;
ttl_str = sprintf('Un-stable $\\phi$');
title(ttl_str,'Interpreter','latex','FontSize',20)

% Stable eigenfunction
subplot(1,2,2)
surf_vals2 = phiStable_PI;
s2 = pcolor(xx1,xx2,surf_vals2);hold on
set(s2,'Edgecolor','none')
colorbar
set(gca,'ColorScale','log')
%plot contour
ic_pts = 1;
l = streamslice(X,Y,u,v); hold on;
set(l,'LineWidth',1)
set(l,'Color','k');
f = @(t,x)[x(2); -delta*x(2) - x(1)^3 + x(1)]; 
bounds = domain(2);
xl = -bounds; xh = bounds;
yl = -bounds; yh = bounds;
for x0 = linspace(-bounds, bounds, ic_pts)
    for y0 = linspace(-bounds, bounds, ic_pts)
        [ts,xs] = ode45(@(t,x)f(t,x),tspan,[x0 y0]);
        plot(xs(:,1),xs(:,2),'k','LineWidth',1); hold on;
    end
end

xlim([-3,3])
ylim([-3,3])
axes = gca;
axis square
set(axes,'FontSize',15);
xlabel('$x_1$','FontSize',20, 'Interpreter','latex')
ylabel('$x_2$','FontSize',20, 'Interpreter','latex')
zlabel('$\phi(x)$','FontSize',20, 'Interpreter','latex')
box on
axes.LineWidth=2;
% colorbar
ttl_str = sprintf('Stable $\\phi$');
title(ttl_str,'Interpreter','latex','FontSize',20)
% legend_strs = {'PI','Analytical'};
% legend(legend_strs,'Interpreter','latex','location','best')
% view(2)


%% Helper Functions
function [value, isterminal, direction] = offFrame(~, Y, boundary)
    value = (norm(Y) > boundary) | (norm(Y) < 1e-6);
    isterminal = 1;
    direction = 0;
end

function integralVal = getIntegral(t,y,ev,wfn)
if ev>0
    iTraj2 = exp(-t(:)*ev).*wfn(y(:,1),y(:,2));
    integralVal = trapz(t,iTraj2,1);
elseif ev<0
    iTraj2 = exp(abs(t(:))*ev).*wfn(y(:,1),y(:,2));
    integralVal = trapz(t,iTraj2,1);
end
end