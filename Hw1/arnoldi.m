function [Q,H]=arnoldi(A,b,m)
% [Q,H]=arnoldi(A,b,m)
% A simple implementation of the Arnoldi method.
% The algorithm will return an Arnoldi "factorization":
%   Q*H(1:m+1,1:m)-A*Q(:,1:m)=0
% where Q is an orthogonal basis of the Krylov subspace
% and H a Hessenberg matrix.
%
% Example:
%  A=randn(100); b=randn(100,1);
%  m=10;
%  [Q,H]=arnoldi(A,b,m);
%  should_be_zero1=norm(Q*H-A*Q(:,1:m))
%  should_be_zero2=norm(Q'*Q-eye(m+1))
    n=length(b);;
    Q=zeros(n,m+1);
    Q(:,1)=b/norm(b);
    
    t0=0;
    t1=0;
    
    for k=1:m
        w=A*Q(:,k); % Matrix-vector product 
                    % with last element        
        %%% Orthogonalize w against columns of Q
       % replace this with a orthogonalization 
        [h,beta,worth]=hw1_good_gs(Q,w,k);
        %%% Put Gram-Schmidt coefficients into H
        H(1:(k+1),k)=[h;beta]; 
        %%% normalize
        Q(:,k+1)=worth/beta; 
    end
end

rand(’seed’,0); 
A=gallery(’wathen’,nn,nn); 
[Q,H]=arnoldi(A,randn(nn,1),10);