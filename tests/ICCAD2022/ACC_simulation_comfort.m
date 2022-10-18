%% Results from PYCASSE
% xe__0 0
% ve__0 0
% xl__0 50
% vl__0 0
% p 0.99375
% c -7.5
% b_a 5
% b_g 0.348809
% The set of optimal parameter values is {'p': 0.99375, 'c': -7.5} with the cost -91.875 and the robustness estimate 0.3488093706746431.
% Time elaspsed for MILP: 444.9479172229767 [seconds].

%% MATLAB Simulation
% Clear workspace 
clf

% Initialize constants
simu_T = 10;
delta_t = 0.5;
H = simu_T/delta_t;
d_safe = 10;
sigma_al = 0.5;
thres = -7.5;
M = 10^5;

% Define System Dynamics
K = 0.5;
tau = 1.6;
A = [1 delta_t 0 0; 0 1 0 0; 0 0 1 delta_t; 0 0 0 1];
B = [0; delta_t; 0; 0];
C = [-1 0 1 0; 0 -1 0 1; 0 1 0 0];
D = [K K -tau*K];
E = -d_safe*K;
w_mean = [0; 0; 0; 0];
Q = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 (sigma_al*delta_t)^2];
v_mean = [0; 0; 0];
R = [1^2 0 0; 0 1^2 0; 0 0 0.5^2];

ae_hist = zeros(1,H+1);
count = 0;

% Initialize figure
figure(1)
hold on
grid on
plot(inf, 'g')
plot(inf, 'r')
legend('satisfying','non-satisfying','AutoUpdate','off')
plot(0:H, thres.*ones(1,H+1), 'b');
ax = gca;
ax.FontSize = 14;
xlabel('Time [s]', 'FontSize', 14);
xlim([0 simu_T])
ylabel('a_e [m/s^2]', 'FontSize', 14)
% ylim([30 80]);

% Simulate
tic
for j = 1:M
    for i = 0:H
        if i == 0
            x = [0; 0; 50; 0];
%             x = [0; 30; 58; 25];
            z = C*x + mvnrnd(v_mean, R)';
            u = D*z + E;
        else
            w = mvnrnd(w_mean, Q)';
            x = A*x + B*u + w;
            z = C*x + mvnrnd(v_mean, R)';
            u = D*z + E;
        end

    %     if x(2) > 30
    %         u = 0;
    %     elseif x(2)+delta_t*u > 30
    %         u = (30-x(2))/delta_t;
    %     end
    %     
    %     if u > 2
    %         u = 2;
    %     elseif u < -3
    %         u = -3;
    %     end

        ae_hist(i+1) = u(1);
    end
    
    if min(ae_hist) < thres
        plot(0:delta_t:simu_T, ae_hist, 'r');
        count = count + 1;
    else
        plot(0:delta_t:simu_T, ae_hist, 'g');
    end
end
hold off
toc