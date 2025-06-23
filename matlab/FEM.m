clear all
close all

function isM = isMmatrix(A)
d = diag(A);
if any(d <= 0)
    isM = false;
    return;
end

% Get off-diagonal nonzeros
[i, j, v] = find(A);
offdiag = i ~= j;

if any(v(offdiag) > 0)
    isM = false;
    return;
end

% Passed both tests
isM = true;
end

T = 35;

for Sigma_d = [10, 1, 0.1]
    for dt = [0.1 0.05 0.025]
        for ne = [64 128 256]
            fprintf("\nSigma_d = %f | dt = %f | ne = %i\n", Sigma_d, dt, ne)

            nt = round(T / dt);


            h = 1 / (ne - 1);
            x = linspace(0,1,ne);
            y = linspace(0,1,ne);
            [X, Y] = meshgrid(x, y);

            % Initial condition: u0 = 1 in upper-right corner
            u0 = double(X >= 0.9 & Y >= 0.9);
            u = u0(:);

            % Reaction term f(u)
            ft = 0.2383;
            fr = 0;
            fd = 1;

            f = @(u) 18.515 * (u - fr) .* (u - ft) .* (u - fd);

            % Diffusivity values
            Sigma_h = 9.5298e-4;

            % Assemble mass matrix
            M = assembleMass(ne, ne, h, h);

            % Compute element centers for diffusivity assignment
            cx = 0.5 * (X(1:end-1,1:end-1) + X(2:end,2:end));
            cy = 0.5 * (Y(1:end-1,1:end-1) + Y(2:end,2:end));
            cx = cx(:);
            cy = cy(:);
            ce = numel(cx);
            sigma = Sigma_h * ones(ce, 1);


            % Diseased regions
            sigma((cx - 0.3).^2 + (cy - 0.7).^2 < 0.1^2)  = Sigma_d * Sigma_h;
            sigma((cx - 0.7).^2 + (cy - 0.3).^2 < 0.15^2) = Sigma_d * Sigma_h;
            sigma((cx - 0.5).^2 + (cy - 0.5).^2 < 0.1^2)  = Sigma_d * Sigma_h;

            % Assemble diffusion matrix with element-wise diffusivity
            A = assembleDiffusion(ne, ne, h, h, sigma);

            % IMEX matrices
            LHS = M + dt * A; % (I - dt * diffusion)
            [~, R] = chol(LHS);
            if R > 0
                error('LHS matrix not SPD.');
            end

            % Initialize debug plot
            figure;
            hImg = imagesc(x, y, reshape(u, ne, ne), [0 1]);  % consistent color limits
            axis xy equal tight;
            colormap turbo;
            colorbar;
            title(sprintf('u(x,y,t = %.2f)', 0));
            drawnow;

            inbounds = true;
            for it = 1:nt
                rhs = M * u - dt * (M * f(u));
                u = LHS \ rhs;

                % Check bounds
                if any(u < -1e-6) || any(u > 1 + 1e-6)
                    inbounds = false;
                end

                % Display every 10 steps or on important frames
                if mod(it, 10) == 0 || it == 1 || it == nt
                    set(hImg, 'CData', reshape(u, ne, ne));
                    title(sprintf('u(x,y,t = %.2f)', it * dt));
                    drawnow;
                    pause(0.1);
                end

                % Check if u > ft everywhere
                if all(u > ft)
                    fprintf("%.3f & %i & %.3f & %s & ", dt, ne, it*dt, string(isMmatrix(LHS)));
                    break;
                end
            end

            fprintf("%s \\\\ \n", string(inbounds));
        end
    end
end
