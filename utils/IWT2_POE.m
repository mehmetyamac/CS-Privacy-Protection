function x = IWT2_POE(wc,L,qmf)
% IWT2_POE -- Inverse  Wavelet Transform 2-D Periodic, Orthogonal,  Extended
%  Usage
%    x = IWT2_POE(wc,L,qmf)
%  Inputs
%    wc    2-d wavelet transform [m by n array; m, n powers of 2]
%    L     coarse level
%    qmf   quadrature mirror filter
%  Outputs
%    x     2-d signal reconstructed from wc
%
%  Description
%    If wc is the result of a forward 2d wavelet transform, with
%    wc = FWT2_POE(x,L,qmf), then x = IWT2_POE(wc,L,qmf) reconstructs x
%    exactly (if qmf is nice).
%
%  See Also
%    FWT2_PO, IWT2_PO, FWT2_POE, MakeONFilter
%
	[m, n, Jm, Jn] = pow2length(wc);
	x = wc;
    J = min(Jm, Jn);
    nr = 2^(Jm - J + L + 1);  nc = 2^(Jn - J + L + 1);
	for jscal= L : J-1,
		topr = (nr/2+1):nr; botr = 1:(nr/2);  allr = 1:nr;
		for iy=1:nc,
			x(allr,iy) =  UpDyadLo(x(botr,iy)',qmf)'  ...
					   + UpDyadHi(x(topr,iy)',qmf)'; 
		end
        topc = (nc/2+1):nc; botc = 1:(nc/2); allc = 1:nc;
		for ix=1:nr,
			x(ix,allc) = UpDyadLo(x(ix,botc),qmf)  ... 
					  + UpDyadHi(x(ix,topc),qmf);
		end
		nc = 2*nc; nr = 2*nr;
	end	
%
% Copyright (c) 1993. David L. Donoho
% Extension to the case m !=n, Brani Vidakovic, 2003.         
%   
% Part of WaveLab Version 802
% Built Sunday, October 3, 1999 8:52:27 AM
% This is Copyrighted Material
% For Copying permissions see COPYING.m
% Comments? e-mail wavelab@stat.stanford.edu
%   
    
