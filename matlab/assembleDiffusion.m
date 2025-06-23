function A = assembleDiffusion(nvx, nvy, hx, hy, diffusivity)
%%
%% nvx         number of vertices along x
%% nvy         number of vertices along y
%% hx          mesh size along x
%% hy          mesh size along y
%% diffusivity vector of size number of elements (ne x 1)

% reference diffusion matrix
Aref = [1 -1; -1 1];
% reference mass matrix
Mref = [1/3 1/6; 1/6 1/3];

% 1d diffusion matrix along x
Ax = 1/hx * Aref;
% 1d diffusion matrix along y
Ay = 1/hy * Aref;

% 1d mass matrix along x
Mx = hx * Mref;
% 1d mass matrix along y
My = hy * Mref;

% local stiffness matrix
Aloc = kron(My, Ax) + kron(Ay, Mx);

% number of vertices and elements
nv = nvx * nvy;
ne = (nvx - 1) * (nvy - 1);

% create connectivity matrix
id = reshape(1:nv, nvx, nvy);

a = id(1:end-1, 1:end-1); a = a(:)';
b = id(2:end,   1:end-1); b = b(:)';
c = id(1:end-1, 2:end);   c = c(:)';
d = id(2:end,   2:end);   d = d(:)';

conn = [a; b; c; d];

% create index matrices for assembly
ii = repmat((1:4)', 1, 4);
jj = ii';

I = conn(ii(:), :);
J = conn(jj(:), :);

% assemble global matrix using scaled local contributions
V = Aloc(:) * ones(1, ne);
V = V .* diffusivity(:)';
V = V(:);

A = sparse(I, J, V);