function error_avg = rmse(X_pred,X_true,prediction)

n_traj = prediction.n_traj;
n_steps = prediction.n_steps;
n = prediction.n;
dt = prediction.dt;
time_steps = 0:dt:(n_steps-1)*dt;

%% get rmse for each trajectory 
x_err = []; x_err_pcent = [];
for j=1:n_traj

    % partse each traj
    start_idx = (j-1)*n+1;
    end_idx = j*n;
    x_true_j = X_true(:,start_idx:end_idx);
    x_pred_j = X_pred(:,start_idx:end_idx);

    % get rmse and % error for each trajectory
    x_error =x_true_j - x_pred_j;
    x_mse = norm(x_error);
    x_pcent = mean(abs(x_error)./abs(x_true_j)) *100;
    
    % sum rmse for each control
    x_err =  [x_err, x_mse];
    x_err_pcent = [x_err_pcent, x_pcent];
    
    % plot the trajectories if show_plot = true
    if prediction.show_plot
        colors = colororder;
        blue = colors(1,:);
        orange = colors(2,:);
        figure(3)
        nexttile
        plot(x_true_j(:,1),x_true_j(:,2),'o','Color',blue); hold on;
        plot(x_true_j(:,1),x_true_j(:,2),'-','LineWidth',2, 'Color',blue); hold on
        plot(x_pred_j(:,1),x_pred_j(:,2),'x','Color',orange); hold on;
        plot(x_pred_j(:,1),x_pred_j(:,2),'--','LineWidth',2, 'Color',orange); hold on
        axes1 = gca;
        box(axes1,'on');
        set(axes1,'FontSize',15,'LineWidth',2);
        xlabel("$\theta$",'interpreter','latex', 'FontSize', 20)
        ylabel("$\dot{\theta}$",'interpreter','latex', 'FontSize', 20)
        lgd = legend('','True','','Predicted');
        lgd.FontSize = 15;
        lgd.Interpreter = 'latex';
    end
end

% avg rmse for all trajectories
error_avg.rmse = [mean(x_err), std(x_err)];
error_avg.pcent = [mean(x_err_pcent), std(x_err_pcent)];


