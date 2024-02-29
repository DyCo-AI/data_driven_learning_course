function dxdt = dynamics_duffing(t,x)
    delta = 0.5; 
    scaling = 1;
    dxdt = [x(2); + x(1) - delta*x(2) - scaling.*x(1)^3]; 
end
