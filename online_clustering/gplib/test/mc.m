function [mf, Sigm]  = mc(pivec, K, nutilde)

nout = size(K, 3);
n = size(K, 1);
B=zeros(n,n,nout);
cholA=zeros(n,n,nout);
for k1=1:nout
    Dsq=sqrt(pivec((1:n)+n*(k1-1))); % Dsq = diag( D^(1/2) )
    A=(Dsq*Dsq').*K(:,:,k1); % = D^(1/2) K D^(1/2)
    A(1:n+1:end)=A(1:n+1:end)+1;
    cholA(:,:,k1)=chol(A,'lower');
    invcholADsq=cholA(:,:,k1)\diag(Dsq);
    B(:,:,k1)=invcholADsq'*invcholADsq;
end
cholP=chol(sum(B,3),'lower') % = chol( R^T B R )

Knu = zeros(n*nout,1);
BKnu = zeros(n,nout);
BK = zeros(n,n,nout);

% update posterior mean
for k1=1:nout
    Knu((1:n)+n*(k1-1))=K(:,:,k1)*nutilde(:,k1);
    BKnu(:,k1)=B(:,:,k1)*Knu((1:n)+(k1-1)*n);
    BK(:,:,k1)=B(:,:,k1)*K(:,:,k1);
    invcholPBK(:,:,k1)=cholP\BK(:,:,k1);
end
invPBKnu=cholP'\(cholP\sum(BKnu,2));
for k1=1:nout
    mf((1:n)+(k1-1)*n)=Knu((1:n)+(k1-1)*n)-K(:,:,k1)*(BKnu(:,k1)-B(:,:,k1)*invPBKnu);
end

% update posterior covariance
for k1=1:nout
    Sigm(k1,k1,:)=diag(K(:,:,k1))-sum(K(:,:,k1).*BK(:,:,k1))'+sum(invcholPBK(:,:,k1).*invcholPBK(:,:,k1))';
    for j1=(k1+1):nout
        Sigm(k1,j1,:)=sum(invcholPBK(:,:,k1).*invcholPBK(:,:,j1));
        Sigm(j1,k1,:)=Sigm(k1,j1,:);
    end
end
              
