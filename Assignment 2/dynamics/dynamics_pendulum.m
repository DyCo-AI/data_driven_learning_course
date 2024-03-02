function dxdt = dynamics_pendulum(t,x)
    g = 9.81;
    L = 1;
    damping_ratio = 2;
    theta = x(1);
    theta_dot = x(2); 
    dxdt = [theta_dot; -(g/L).*sin(theta)-damping_ratio*theta_dot];
end