function wc = FWT2_POE(x,L,qmf)
% FWT2_POE -- Forward Wavelet Transform 2-D Periodic, Orthogonal,  Extended
% 
%  Usage
%    wc = FWT2_POE(x,L,qmf)
%  Inputs
%    x     2-d image (m by n array; m, n dyadic)
%    L     coarse level
%    qmf   quadrature mirror filter
%  Outputs
%    wc    2-d wavelet transform
%
%  Description
%    A two-dimensional Wavelet Transform is computed for the
%    array x.  To reconstruct, use IWT2_POE.
%
%  See Also
%    FWT2_PO, IWT_PO2, IWT2_POE, MakeONFilter
%
	[m, n, Jm, Jn] = pow2length(x);
	wc = x; 
    nr = m;
	nc = n;
    J=min(Jm, Jn);
	for jscal=J-1:-1:L,
		topc = (nc/2+1):nc; botc = 1:(nc/2);
		for ix=1:nr,
			row = wc(ix,1:nc);
			wc(ix,botc) = DownDyadLo(row,qmf);
			wc(ix,topc) = DownDyadHi(row,qmf);
		end
        topr = (nr/2+1):nr; botr = 1:(nr/2);
		for iy=1:nc,
			row = wc(1:nr,iy)';
			wc(topr,iy) = DownDyadHi(row,qmf)';
			wc(botr,iy) = DownDyadLo(row,qmf)'; 
		 end
		nc = nc/2;   nr = nr/2;
	end   
 
%
% Copyright (c) 1993. David L. Donoho
% Adapted to the case m != n, Brani Vidakovic  2003.  
    
    
%   
% Part of WaveLab Version 802
% Built Sunday, October 3, 1999 8:52:27 AM
% This is Copyrighted Material
% For Copying permissions see COPYING.m
% Comments? e-mail wavelab@stat.stanford.edu
%