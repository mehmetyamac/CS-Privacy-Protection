
function [m, n, Jm, Jn] = pow2length(x)
% pow2length -- Find size and dyadic length of a matrix
%  Usage
%    [m, n, Jm, Jn] = pow2length(x)
%  Inputs
%    x   2-d image of size m x n;  m=2^Jm, n = 2^Jn (hopefully)
%  Outputs
%    m, n   dimensions of x
%    Jm, Jn   least power of two greater or equal than m, n
%
%  Side Effects
%    A warning message is issued if either m or n are not powers of 2,
%    
%
	s = size(x);
	m = s(1);
    n = s(2);
	km = 1 ; Jm = 0; while km < m , km=2*km; Jm = 1+Jm ; end ;
	if km ~= m ,
		  disp('Warning in quadlength: m != 2^Jm')
	end
	kn = 1 ; Jn = 0; while kn < n , kn=2*kn; Jn = 1+Jn ; end ;
	if kn ~= n ,
		  disp('Warning in quadlength: n != 2^Jn')
	end

%
% Copyright (c) 1993. David L. Donoho
% Adapted to the case m != n Brani Vidakovic 2003.    
