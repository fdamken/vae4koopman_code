import numpy as np

# Returns: net
from src.orig.kalmansmooth import kalmansmooth



def lds(X, K = 2, T = None, cyc = 100, tol = 0.0001):  # function net=lds(X,K,T,cyc,tol);
    p = len(X[0, :])  # p=length(X(1,:));
    N = len(X[:, 0])  # N=length(X(:,1));
    if T is None:  # nargin stuff
        T = N

    Mu = np.mean(X, axis = 0).reshape(1, -1)  # Mu=mean(X);
    X = X - np.ones((N, 1)) @ Mu  # X=X-ones(N,1)*Mu;

    if N % T != 0:  # if (rem(N,T)~=0)
        # disp('Error: Data matrix length must be multiple of sequence length T');
        print('Error: Data matrix length must be multiple of sequence length T')
        return  # return;
    # end;

    N = N / T  # N=N/T;

    # TODO: Replace to support other state/measurement sizes.
    A = np.eye(3)  # A = [1 0 0; 0 1 0; 0 0 1];
    Q = np.ones(3)  # Q = [1 0 0; 0 1 0; 0 0 1];
    C = np.eye(3)  # C = [1 0 0; 0 1 0; 0 0 1];
    R = np.ones(3)  # R = [1 0 0; 0 1 0; 0 0 1];
    x0 = np.zeros(3).reshape(-1, 1)  # x0 = [0; 0; 0];
    P0 = np.eye(3)  # P0 = [1 0 0; 0 1 0; 0 0 1];

    lik = 0  # lik=0;
    LL = []  # LL=[];

    Y = X.reshape(int(T), int(N), int(p))  # Y=reshape(X,T,N,p);
    Y = np.transpose(Y, axes = [1, 2, 0])  # Y=permute(Y,[2 3 1]); % Y is (N,p,T), analogously to X

    YY = np.sum(np.multiply(X, X), axis = 0) / (T * N)  # YY=sum(X.*X)'/(T*N);

    for cycle in range(cyc):  # for cycle=1:cyc
        # E STEP
        oldlik = lik  # oldlik=lik;
        lik, Xfin, Pfin, Ptsum, YX, A1, A2, A3 = kalmansmooth(A, C, Q, R, x0, P0, Y)  # [lik,Xfin,Pfin,Ptsum,YX,A1,A2,A3]=kalmansmooth(A,C,Q,R,x0,P0,Y);
        LL.append(lik)  # LL=[LL lik];
        print('cycle %d lik %f' % (cycle, lik))  # fprintf('cycle %g lik %g',cycle,lik);

        if cycle <= 2:  # if (cycle<=2)
            likbase = lik  # likbase=lik;
        elif lik < oldlik:  # elseif (lik<oldlik)
            print('violation')  # fprintf(' violation');
        elif (lik - likbase) < (1 + tol) * (oldlik - likbase) or not np.isfinite(lik):  # elseif ((lik-likbase)<(1 + tol)*(oldlik-likbase)|~isfinite(lik))
            print()  # fprintf('\n');
            break  # break;
        # end;
        print()  # fprintf('\n');

        # M STEP
        x0 = np.sum(Xfin[:, :, 0], axis = 0).reshape(1, -1).T / N  # x0=sum(Xfin(:,:,1),1)'/N;
        T1 = Xfin[:, :, 0] - np.ones((int(N), 1)) @ x0.T  # T1=Xfin(:,:,1)-ones(N,1)*x0';
        P0 = Pfin[:, :, 0] + T1.T @ T1 / N  # P0=Pfin(:,:,1)+T1'*T1/N;
        C = YX @ np.linalg.inv(Ptsum) / N  # C=YX*inv(Ptsum)/N;
        R = YY - np.diag(C @ YX.T) / (T * N)  # R=YY-diag(C*YX')/(T*N);
        A = A1 @ np.linalg.inv(A2)  # A=A1*inv(A2);
        Q = (1 / (T - 1)) * np.diag(np.diag(A3 - A @ A1.T))  # Q=(1/(T-1))*diag(diag((A3-A*(A1'))));
        if np.linalg.det(Q) < 0:  # if (det(Q)<0)
            print('Q problem')  # fprintf('Q problem\n');
        # end;
    # end;

    return { 'A': A, 'C': C, 'Q': Q, 'R': R, 'x0': x0, 'P0': P0, 'Mu': Mu, 'LL': LL }
