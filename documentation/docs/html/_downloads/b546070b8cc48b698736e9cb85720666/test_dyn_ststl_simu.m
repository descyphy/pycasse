%% MATLAB Simulation
% Clear workspace 
clf

% Initialize constants
simu_T = 10;
delta_t = 1;
H = simu_T/delta_t;
thres = 34;
M = 10^5;

% Define System Dynamics
A = [1 delta_t; 0 1];
B = [0; delta_t];
w_mean = [0; 0];
Q = [0 0; 0 (0.5*delta_t)^2];

s_hist = zeros(1,H+1);
count = 0;

% Initialize figure
figure(1)
hold on
grid on
plot(inf, 'g')
plot(inf, 'r')
plot(0:H, thres.*ones(1,H+1), 'b');
ax = gca;
ax.FontSize = 14;
xlabel('Time [s]', 'FontSize', 14);
xlim([0 simu_T])
ylabel('s [m]', 'FontSize', 14)

% Simulate
tic
for j = 1:M
    for i = 0:H
        if i == 0
            x = [0; 0];
            u = 1;
        else
            w = mvnrnd(w_mean, Q)';
            x = A*x + B*u + w;
            if i >= 9
                u = -1;
            else
                u = 1;
            end
        end

        s_hist(i+1) = x(1);
    end
    
    if max(s_hist) < thres
        plot(0:delta_t:simu_T, s_hist, 'r');
        count = count + 1;
    else
        plot(0:delta_t:simu_T, s_hist, 'g');
    end
end
legend("Satisfying Percentage: " + (M-count)/M, "Non-Satisfying Percentage: " + count/M, 'AutoUpdate','off')
hold off
toc