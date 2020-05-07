import numpy as np



# Returns: lik, Xfin, Pfin, Ptsum, YX, A1, A2, A3
def kalmansmooth(A, C, Q, R, x0, P0, Y):  # function  [lik,Xfin,Pfin,Ptsum,YX,A1,A2,A3]=kalmansmooth(A,C,Q,R,x0,P0,Y);
    (N, p, T) = Y.shape  # [N p T]=size(Y);
    K = len(x0)  # K=length(x0);
    tiny = np.exp(-700)  # tiny=exp(-700);
    I = np.eye(K)  # I=eye(K);
    const = (2 * np.pi) ** (-p / 2)  # const=(2*pi)^(-p/2);
    problem = 0  # problem=0;
    lik = 0  # lik=0;

    Xpre = np.zeros((N, K))  # Xpre=zeros(N,K);
    Xcur = np.zeros((N, K, T))  # Xcur=zeros(N,K,T);
    Xfin = np.zeros((N, K, T))  # Xfin=zeros(N,K,T);

    Ppre = np.zeros((K, K, T))  # Ppre=zeros(K,K,T);
    Pcur = np.zeros((K, K, T))  # Pcur=zeros(K,K,T);
    Pfin = np.zeros((K, K, T))  # Pfin=zeros(K,K,T);

    Pt = np.zeros((K, K))  # Pt=zeros(K,K);
    Pcov = np.zeros((K, K))  # Pcov=zeros(K,K);
    Kcur = np.zeros((K, p))  # Kcur=zeros(K,p);
    invP = np.zeros((p, p))  # invP=zeros(p,p);
    J = np.zeros((K, K, T))  # J=zeros(K,K,T);

    #
    # FORWARD PASS

    R = R + (R == 0) * tiny  # R=R+(R==0)*tiny;

    Xpre = np.ones((N, 1)) @ x0.T  # Xpre=ones(N,1)*x0';
    Ppre[:, :, 0] = P0  # Ppre(:,:,1)=P0;
    invR = np.diag(1 / R)  # invR=diag(1./R);

    for t in range(0, T):  # for t=1:T
        if K < p:  # if (K<p)
            temp1 = C / R.reshape(-1, 1)  # temp1=rdiv(C,R);
            temp2 = temp1 @ Ppre[:, :, t]  # temp2=temp1*Ppre(:,:,t); % inv(R)*C*Ppre
            temp3 = C.T @ temp2  # temp3=C'*temp2;
            temp4 = np.linalg.inv(I + temp3) @ temp1.T  # temp4=inv(I+temp3)*temp1';
            invP = invR - temp2 @ temp4  # invP=invR-temp2*temp4;
            CP = temp1.T - temp3 @ temp4  # CP= temp1' - temp3*temp4;  % C'*invP
        else:  # else
            temp1 = np.diag(R) + C @ Ppre[:, :, t] @ C.T  # temp1=diag(R)+C*Ppre(:,:,t)*C';
            invP = np.linalg.inv(temp1)  # invP=inv(temp1);
            CP = C.T @ invP  # CP=C'*invP;
        # end;

        Kcur = Ppre[:, :, t] @ CP  # Kcur=Ppre(:,:,t)*CP;
        KC = Kcur @ C  # KC=Kcur*C;
        Ydiff = Y[:, :, t] - Xpre @ C.T  # Ydiff=Y(:,:,t)-Xpre*C';
        Xcur[:, :, t] = Xpre + Ydiff @ Kcur.T  # Xcur(:,:,t)=Xpre+Ydiff*Kcur';
        Pcur[:, :, t] = Ppre[:, :, t] - KC @ Ppre[:, :, t]  # Pcur(:,:,t)=Ppre(:,:,t)-KC*Ppre(:,:,t);
        if t < T - 1:  # if (t<T)
            Xpre = Xcur[:, :, t] @ A.T  # Xpre=Xcur(:,:,t)*A';
            Ppre[:, :, t + 1] = A @ Pcur[:, :, t] @ A.T + Q  # Ppre(:,:,t+1)=A*Pcur(:,:,t)*A'+Q;
        # end;

        # calculate likelihood
        detiP = np.sqrt(np.linalg.det(invP))  # detiP=sqrt(det(invP));
        if np.isreal(detiP) and detiP > 0:  # if (isreal(detiP) & detiP>0)
            # Use axis=0
            lik = lik + N * np.log(detiP) - 0.5 * np.sum(np.sum(np.multiply(Ydiff, Ydiff @ invP), axis = 0), axis = 0)  # lik=lik+N*log(detiP)-0.5*sum(sum(Ydiff.*(Ydiff*invP)));
        else:  # else
            problem = 1  # problem=1;
        # end;
    # end;

    lik = lik + N * T * np.log(const)  # lik=lik+N*T*log(const);

    #
    # BACKWARD PASS

    A1 = np.zeros((K, K))  # A1=zeros(K);
    A2 = np.zeros((K, K))  # A2=zeros(K);
    A3 = np.zeros((K, K))  # A3=zeros(K);
    Ptsum = np.zeros((K, K))  # Ptsum=zeros(K);
    YX = np.zeros((p, K))  # YX=zeros(p,K);

    t = T - 1  # t=T;
    Xfin[:, :, t] = Xcur[:, :, t]  # Xfin(:,:,t)=Xcur(:,:,t);
    Pfin[:, :, t] = Pcur[:, :, t]  # Pfin(:,:,t)=Pcur(:,:,t);
    Pt = Pfin[:, :, t] + Xfin[:, :, t].T @ Xfin[:, :, t] / N  # Pt=Pfin(:,:,t) + Xfin(:,:,t)'*Xfin(:,:,t)/N;
    A2 = -Pt  # A2= -Pt;
    Ptsum = Pt  # Ptsum=Pt;

    YX = Y[:, :, t].T @ Xfin[:, :, t]  # YX=Y(:,:,t)'*Xfin(:,:,t);

    for t in reversed(range(0, T - 1)):  # for t=(T-1):-1:1
        J[:, :, t] = Pcur[:, :, t] @ A.T @ np.linalg.inv(Ppre[:, :, t + 1])  # J(:,:,t)=Pcur(:,:,t)*A'*inv(Ppre(:,:,t+1));
        Xfin[:, :, t] = Xcur[:, :, t] + (Xfin[:, :, t + 1] - Xcur[:, :, t] @ A.T) @ J[:, :, t].T  # Xfin(:,:,t)=Xcur(:,:,t)+(Xfin(:,:,t+1)-Xcur(:,:,t)*A')*J(:,:,t)';
        # Pfin(:,:,t)=Pcur(:,:,t)+J(:,:,t)*(Pfin(:,:,t+1)-Ppre(:,:,t+1))*J(:,:,t)';
        Pfin[:, :, t] = Pcur[:, :, t] + J[:, :, t] @ (Pfin[:, :, t + 1] - Ppre[:, :, t + 1]) @ J[:, :, t].T
        Pt = Pfin[:, :, t] + Xfin[:, :, t].T @ Xfin[:, :, t] / N  # Pt=Pfin(:,:,t) + Xfin(:,:,t)'*Xfin(:,:,t)/N;
        Ptsum = Ptsum + Pt  # Ptsum=Ptsum+Pt;
        YX = YX + Y[:, :, t].T @ Xfin[:, :, t]  # YX=YX+Y(:,:,t)'*Xfin(:,:,t);
    # end;

    A3 = Ptsum - Pt  # A3= Ptsum-Pt;
    A2 = Ptsum + A2  # A2= Ptsum+A2;

    t = T - 1  # t=T;
    Pcov = (I - KC) @ A @ Pcur[:, :, t - 1]  # Pcov=(I-KC)*A*Pcur(:,:,t-1);
    A1 = A1 + Pcov + Xfin[:, :, t].T @ Xfin[:, :, t - 1] / N  # A1=A1+Pcov+Xfin(:,:,t)'*Xfin(:,:,t-1)/N;

    for t in reversed(range(1, T - 1)):  # for t=(T-1):-1:2
        Pcov = (Pcur[:, :, t] + J[:, :, t] @ (Pcov - A @ Pcur[:, :, t])) @ J[:, :, t - 1].T  # Pcov=(Pcur(:,:,t)+J(:,:,t)*(Pcov-A*Pcur(:,:,t)))*J(:,:,t-1)';
        A1 = A1 + Pcov + Xfin[:, :, t].T @ Xfin[:, :, t - 1] / N  # A1=A1+Pcov+Xfin(:,:,t)'*Xfin(:,:,t-1)/N;
    # end;

    if problem:  # if problem
        print('problem')  # fprintf(' problem  ');
        problem = 0  # problem=0;
    # end;

    return lik, Xfin, Pfin, Ptsum, YX, A1, A2, A3
