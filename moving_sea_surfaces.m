clear;close;clc;
%% ----------------------------- Parameters ---------------------------------
rng(2025);
% Radar parameters
freq   = 9.39e9;  % Center frequency (Hz)
prf    = 500;     % PRF (Hz)
tpd    = 100e-6;  % Pulse width (s)
azBw   = 2;       % Azimuth 3-dB beamwidth (deg)
depang = 30;      % Depression angle (deg)
antGain= 45.7;    % Not used in this simplified chain
fs     = 10e6;    % Sampling frequency (Hz)
bw     = 5e6;     % LFM sweep bandwidth (Hz)

% Sea surface parameters
seaState   = 4;     % 1..7
wind_knots = 19;    % knots
L          = 512;   % side length (m)
resSurface = 2;     % grid spacing (m)

% Conversions / derived
[c,lambda] = deal(physconst_c, physconst_c/freq);
range_res  = c/(2*bw);           % (~30 m)
azBW_rad   = deg2rad(azBw);

% Simulation length
numPulses = 1024;
T_PRI     = 1/prf;

% Radar geometry (monostatic, fixed platform)
rdrht  = 100;                    % radar height (m)
rdrpos = [-L/2, 0, rdrht];       % place at x=-L/2, y=0
R0     = rdrht/sind(depang);     % boresight slant range to sea

% Wind
wind_ms = 0.514444*wind_knots;   % m/s
winddir = 0;                     % toward +x

%% ---------------------------- Sea synthesis --------------------------------
% z(x,y,t) via directional spectrum synthesis (Phillips/Tessendorf-like)

x = -L/2:resSurface:(L/2-resSurface);
y = -L/2:resSurface:(L/2-resSurface);
[XX,YY] = meshgrid(x,y);
Nx = numel(x); Ny = numel(y);

% k-domain grid
Lx=L; Ly=L;
dkx = 2*pi/Lx; dky = 2*pi/Ly;
kx = ifftshift((-floor(Nx/2):ceil(Nx/2)-1)*dkx);
ky = ifftshift((-floor(Ny/2):ceil(Ny/2)-1)*dky);
[KX,KY] = meshgrid(kx,ky);
K  = sqrt(KX.^2 + KY.^2)+eps;

% Direction cosine to wind
thW = deg2rad(winddir);
wx = cos(thW); wy = sin(thW);
cosTheta = (KX.*wx + KY.*wy)./K; % cos(angle to wind)
cosTheta = max(cosTheta,0);      % upwind only

% Phillips-style spectrum parameters
A = sea_calibrate_A(seaState,wind_ms); 
Lww = wind_ms^2/9.81;            % wavelength scale L (Tessendorf)
kp = 1/Lww;                      % peak wavenumber

% Base omnidirectional spectrum ~ A * exp(-1/(k^2 L^2)) / k^4
S0 = A .* exp(-1./(K.^2 * Lww^2)) ./ (K.^4 + (1/L)^4);

% Directional spreading ~ cos^s(theta)
spreading_s = 20*(wind_ms/10)^(1/2); 
Dtheta = cosTheta.^spreading_s;

% High-k damping (capillary tail rolloff)
kh = 20*kp; tail = exp(-(K/kh).^2);

% Final directional 2D spectrum
Phi = S0 .* Dtheta .* tail; Phi(K==0)=0;

% Random complex amplitudes with Hermitian symmetry for real surface
rng(1);
phi_rand = 2*pi*rand(Ny,Nx);
H0 = (sqrt(Phi/2) .* (cos(phi_rand) + 1i*sin(phi_rand))) * sqrt(dkx*dky);
H0 = imposeHermitian(H0);

% Deep water dispersion (includes small capillary correction)
gamma = 0.074/ (1000*9.81);
omega = sqrt(9.81*K + gamma*(K.^3));
omega(K==0)=0;

% Surface at time t
getZ = @(t) real(ifft2(H0 .* exp(1i*omega*t)))*Nx*Ny;

% Snapshot at t=0
z0 = getZ(0);

% Visualize sea and radar
vizSeaSurfacePlot(x,y,z0,rdrpos);

% CDF of elevation
vizSeaSurfaceCDF(z0(:));

% Significant wave height
actSigHgt = vizEstimateSWH(x,y,z0(:)); 

%% ---------------------------- Radar channel --------------------------------
% Transmit baseband LFM waveform (complex)
Ntx = round(tpd*fs);
tt  = (0:Ntx-1)/fs; B = bw; beta = B/tpd;
s_tx = exp(1i*pi*beta*(tt - tpd/2).^2);

% Matched filter (time-reversed conjugate)
h_mf = conj(flip(s_tx));

% PRI fast-time grid & range grid
numSamples = round(T_PRI*fs);
Trng = (0:numSamples-1)/fs;          % seconds within PRI
rngGrid = time2range_base(Trng,c);   % meters
maxRange = 20e3;                      % truncate
[~,idxTruncate] = min(abs(rngGrid - maxRange));
rngGrid = rngGrid(1:idxTruncate);

% Beam footprint on sea (mainlobe stripe around boresight)
Xsea = XX; Ysea = YY;
Xrel = Xsea - rdrpos(1); Yrel = Ysea - rdrpos(2);
Rapprox = sqrt(Xrel.^2 + Yrel.^2 + rdrht^2);
boresight_mask = (abs(atan2(Yrel,Xrel)) < azBW_rad/2) & ...
                 (abs(Rapprox - R0) < 2*range_res) & ...
                 (Xrel > 0);
idx_pts = find(boresight_mask);

% Downselect scatterers for speed
maxScat = 6000;
if numel(idx_pts) > maxScat
    idx_pts = idx_pts(randperm(numel(idx_pts),maxScat));
end
[Xs,Ys] = deal(Xsea(idx_pts), Ysea(idx_pts));

% Constant NRCS-like weights. Include 1/R^2 spreading & speckle.
sigma0 = db2mag_base(-5);  % clutter reflectivity per patch (linear)
dA     = resSurface^2;      % patch area

% Datacube (raw IQ before matched filter) with small noise floor
noiseSigma = db2mag_base(-80);
iqsig = (randn(idxTruncate,numPulses) + 1i*randn(idxTruncate,numPulses))/sqrt(2) * noiseSigma;

% Main loop over pulses
radar_xyz = rdrpos;
for ip = 1:numPulses
    t_now = (ip-1)*T_PRI;
    Zt = getZ(t_now);
    z_now = interp2(x,y, Zt, Xs, Ys, 'linear', 0); % elevation

    P = [Xs, Ys, z_now];
    dxyz = P - radar_xyz;
    R = sqrt(sum(dxyz.^2,2));

    a = (randn(size(R)) + 1i*randn(size(R)))/sqrt(2);
    a = a .* sqrt(sigma0*dA) ./ (R.^2);

    tau = 2*R/c; s0 = round(tau*fs) + 1;

    for k = 1:numel(s0)
        idx0 = s0(k);
        if idx0 < idxTruncate
            i2 = min(idx0+Ntx-1, idxTruncate);
            nlen = i2-idx0+1;
            iqsig(idx0:i2,ip) = iqsig(idx0:i2,ip) + a(k)*s_tx(1:nlen).';
        end
    end

    if mod(ip,128)==0
        vizRawIQPlot(iqsig); drawnow;
    end
end
vizRawIQPlot(iqsig);

%% ------------------------- Pulse compression --------------------------------
pcResp = zeros(size(iqsig), 'like', iqsig);
for ip = 1:numPulses
    pcResp(:,ip) = conv(iqsig(:,ip), h_mf, 'same');
end

pcsigMagdB = mag2db_base(abs(pcResp));
pcsigMagdB = pcsigMagdB - max(pcsigMagdB(:));

% Range–Time plot (auto color scaling)
vizRangeTimePlot(rngGrid, prf, pcsigMagdB, R0);

% Magnitude vs time at strongest range bin
[~,maxIdx] = max(pcsigMagdB(:));
[idxRange,~] = ind2sub(size(pcsigMagdB),maxIdx);
vizMagTimePlot(pcsigMagdB(idxRange,:),numPulses,prf,rngGrid(idxRange));

%% ------------------------------ STFT ---------------------------------------
[S,F,T] = stft_twosided(pcResp(idxRange,:), prf, 128, 120); % window 128, hop 8
Speed = (F(:)*lambda)/2;
vizSTFTPlot(S,Speed,T,rngGrid(idxRange));

%% ------------------------- Amplitude histogram ------------------------------
[~,idxMin] = min(abs(rngGrid - 180));
[~,idxMax] = min(abs(rngGrid - 210));
subMag = mag2db_base(abs(pcResp(idxMin:idxMax,:)));
subMag = subMag(:); subMag = subMag(~isinf(subMag));
subMag = subMag - min(subMag) + eps; % shift positive
[scale,shape] = weibull_momfit(subMag);
vizHistWeibullPlot(subMag,scale,shape);



%% =========================== Helper functions ===============================
function c = physconst_c
c = 299792458; % m/s
end

function r = time2range_base(t,c)
r = c*t/2; % meters
end

function v = db2mag_base(xdb)
v = 10.^(xdb/20);
end

function ydb = mag2db_base(x)
ydb = 20*log10(abs(x)+eps);
end

function H = imposeHermitian(H)
[Ny,Nx] = size(H);
H = fftshift(H);
Hc = conj(flipud(fliplr(H)));
H(2:end,2:end) = (H(2:end,2:end) + Hc(2:end,2:end))/2;
H = ifftshift(H);
end

function A = sea_calibrate_A(seaState, U10)
base = 1e-3;              % baseline spectral level
scaleW = (U10/10)^2;      % grows ~ U10^2
scaleS = 2^(seaState-4);  % sea state scaling around 4
A = base*scaleW*scaleS;
end

function vizSeaSurfacePlot(x,y,z,rdrpos)
seaColorMap = vizSeaColorMap(256);
figure
z = reshape(z,numel(y),numel(x));
surf(x,y,z) ; hold on
plot3(rdrpos(1),rdrpos(2),rdrpos(3),'ok','LineWidth',2,'MarkerFaceColor','k')
legend('Sea Surface','Radar','Location','Best')
shading interp; axis equal
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Elevations (m)')
stdZ = std(z(:)); minC = -4*stdZ; maxC = 4*stdZ;
minZ = min(minC,rdrpos(3)); maxZ = max(maxC,rdrpos(3));
title('Sea Surface Elevations'); axis tight; zlim([minZ maxZ])
hC = colorbar('southoutside'); colormap(seaColorMap)
hC.Label.String = 'Elevations (m)'; hC.Limits = [minC maxC];
drawnow; pause(0.25)
end

function cmap = vizSeaColorMap(n)
c = hsv2rgb([2/3 1 0.2; 2/3 1 1; 0.5 1 1]);
cmap = zeros(n,3);
cmap(:,1) = interp1(1:3,c(:,1),linspace(1,3,n));
cmap(:,2) = interp1(1:3,c(:,2),linspace(1,3,n));
cmap(:,3) = interp1(1:3,c(:,3),linspace(1,3,n));
end

function vizSeaSurfaceCDF(x)
x = x(~isnan(x)); n = length(x);
x = sort(x(:)); yy = (1:n)'/n; 
notdup = ([diff(x(:)); 1] > 0);
xx = x(notdup); yy = [0; yy(notdup)];
k = length(xx); n2 = reshape(repmat(1:k,2,1),2*k,1);
xCDF = [-Inf; xx(n2); Inf];
yCDF = [0; 0; yy(1+n2)];
figure; plot(xCDF,yCDF,'LineWidth',2); grid on
title('Wave Elevation CDF'); xlabel('Wave Elevation (m)'); ylabel('Probability')
drawnow; pause(0.25)
end

function sigHgt = vizEstimateSWH(x,y,z)
figure
numX = numel(x);
z = reshape(z,numX,numel(y));
zEst = z(floor(numX/2) + 1,:);
plot(x,zEst,'LineWidth',2); grid on; hold on
xlabel('X (m)'); ylabel('Elevation (m)'); title('Wave Height Estimation'); axis tight
idxMin = find_local_extrema(zEst,'min'); idxMax = find_local_extrema(zEst,'max');
co = get(gca,'ColorOrder');
plot(x(idxMin),zEst(idxMin),'v','MarkerFaceColor',co(2,:),'MarkerEdgeColor',co(2,:))
plot(x(idxMax),zEst(idxMax),'^','MarkerFaceColor',co(3,:),'MarkerEdgeColor',co(3,:))
legend('Wave Elevation Data','Trough','Crest','Location','SouthWest')
waveHgts = [];
for ii = 1:numX
    zRow = z(ii,:);
    idxMin = find_local_extrema(zRow,'min'); troughs = zRow(idxMin);
    idxMax = find_local_extrema(zRow,'max'); crests  = zRow(idxMax);
    numWaves = min(numel(troughs),numel(crests));
    if numWaves>0
        waveHgts = [waveHgts, abs(crests(1:numWaves) - troughs(1:numWaves))]; 
    end
end
waveHgts = sort(waveHgts);
if isempty(waveHgts)
    sigHgt = NaN;
else
    idxTopThird = floor(numel(waveHgts)*2/3)+1;
    sigHgt = mean(waveHgts(idxTopThird:end));
end
drawnow; pause(0.25)
end

function idx = find_local_extrema(v,mode)
v = v(:).'; n = numel(v);
idx = false(1,n);
for i=2:n-1
    if strcmp(mode,'min')
        idx(i) = v(i)<v(i-1) && v(i)<v(i+1);
    else
        idx(i) = v(i)>v(i-1) && v(i)>v(i+1);
    end
end
idx = find(idx);
end

function hAx = vizRawIQPlot(iqsig)
figure(567); clf
magdb = mag2db_base(abs(iqsig));
imagesc(magdb); axis xy
hAx = gca;
title('Raw IQ'); xlabel('Pulses'); ylabel('Samples')
hC = colorbar; hC.Label.String = 'Magnitude (dB)';
drawnow; pause(0.25)
end

function vizRangeTimePlot(rngGrid,prf,pcsigMagdB,R0)
figure; numPulses = size(pcsigMagdB,2);
imagesc((1:numPulses)/prf, rngGrid, pcsigMagdB); axis xy
xlabel('Time (sec)'); ylabel('Range (m)')
hC = colorbar; hC.Label.String = 'Magnitude (dB)'; axis tight
if nargin>=4 && ~isempty(R0) && isfinite(R0)
    ylo = max(min(rngGrid), R0-120); yhi = min(max(rngGrid), R0+120);
    ylim([ylo yhi]);
else
    ylim([min(rngGrid) max(rngGrid)]);
end
title('Range versus Time'); drawnow; pause(0.25)
end

function vizMagTimePlot(magVals,numPulses,prf,rngVal)
figure; plot((1:numPulses)/prf, magVals,'LineWidth',2); grid on
xlabel('Time (sec)'); ylabel('Magnitude (dB)'); axis tight
title(sprintf('Magnitude versus Time at Range %.2f (m)',rngVal));
drawnow; pause(0.25)
end

function [S,F,T] = stft_twosided(x, fs, winlen, hop)
x = x(:).'; N = numel(x);
if N < winlen, winlen = N; end
w = 0.5-0.5*cos(2*pi*(0:winlen-1)/(winlen-1));
cols = 1+floor((N-winlen)/hop);
S = zeros(winlen, cols);
T = zeros(1, cols);
for c = 1:cols
    i0 = 1+(c-1)*hop; seg = x(i0:i0+winlen-1).*w;
    S(:,c) = fftshift(fft(seg(:), winlen));
    T(c) = (i0-1 + winlen/2)/fs; % center time
end
F = ((0:winlen-1) - floor(winlen/2))*(fs/winlen);
end

function vizSTFTPlot(S,Speed,T,rngVal)
figure; SdB = mag2db_base(abs(S));
imagesc(T, Speed, SdB); axis xy; colorbar
xlabel('Time (sec)'); ylabel('Speed (m/s)')
title(sprintf('STFT at Range %.2f (m)',rngVal));
drawnow; pause(0.25)
end

function [scale,shape] = weibull_momfit(x)
x = x(:); x = x(x>0);
mx = mean(x); vx = var(x);
cv = sqrt(vx)/mx; if ~isfinite(cv) || cv<=0, shape=2; scale=mx/gamma(1+1/shape); return; end
CVfun = @(k) sqrt(gamma(1+2./k)./gamma(1+1./k).^2 - 1) - cv;
kL=0.5; kU=20; fl=CVfun(kL); fu=CVfun(kU);
if sign(fl)==sign(fu)
    ks = logspace(log10(0.5),log10(50),200);
    vals = arrayfun(CVfun,ks);
    [~,ii]=min(abs(vals)); kstar = ks(ii);
else
    kstar = fzero(CVfun,[kL,kU]);
end
shape = max(real(kstar),0.5);
scale = mx / gamma(1+1/shape);
end

function vizHistWeibullPlot(x,scale,shape)
figure; 
if isempty(x) || ~isfinite(scale) || ~isfinite(shape)
    text(0.5,0.5,'Insufficient data for histogram','HorizontalAlignment','center');
    axis off; return;
end
hold on; grid on
histogram(x,'Normalization','pdf');
xx = linspace(min(x),max(x),300);
pdfW = (shape./scale) .* (xx./scale).^(shape-1) .* exp(- (xx./scale).^shape);
plot(xx,pdfW,'LineWidth',2)
xlabel('Magnitude (dB, shifted)'); ylabel('PDF')
legend('Data','Weibull fit','Location','best')
title(sprintf('Amplitude Histogram & Weibull Fit  (k=%.2f, λ=%.2f)',shape,scale));
end
