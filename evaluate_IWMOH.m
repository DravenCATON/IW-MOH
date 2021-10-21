function [eva,train_time_round] = evaluate_IWMOH(XTrain,LTrain,XQuery,LQuery,Vector,OURparam)
    eva=zeros(1,OURparam.nchunks);
    train_time_round=zeros(1,OURparam.nchunks);
    
    %% preprocess
    XTrain_new=cell2mat(XTrain(1:end,:));
    I_tr=XTrain_new(:,1:OURparam.image_feature_size);
    T_tr=XTrain_new(:,OURparam.image_feature_size+1:end);
    I_te=XQuery(:,1:OURparam.image_feature_size);
    T_te=XQuery(:,OURparam.image_feature_size+1:end);

    Ntrain = size(I_tr,1);
    sample = randsample(Ntrain, OURparam.n_anchors);
    
    anchorI = I_tr(sample,:);
    anchorT = T_tr(sample,:);
    sigmaI = OURparam.sigmaI;
    sigmaT = OURparam.sigmaT;
    
    PhiI = exp(-sqdist(I_tr,anchorI)/(2*sigmaI*sigmaI));
    PhiT = exp(-sqdist(T_tr,anchorT)/(2*sigmaI*sigmaI));
    PhiI(sum(I_tr,2)==0,:)=0;
    PhiT(sum(T_tr,2)==0,:)=0;
    XTra=[PhiI PhiT];

    Phi_testI = exp(-sqdist(I_te,anchorI)/(2*sigmaI*sigmaI));
    Phi_testT = exp(-sqdist(T_te,anchorT)/(2*sigmaT*sigmaT));
    Phi_testI(sum(I_te,2)==0,:)=0;
    Phi_testT(sum(T_te,2)==0,:)=0;
    XQuery=[Phi_testI Phi_testT];

    t=1;
    for chunki = 1:OURparam.nchunks
        chunksize=size(XTrain{chunki,1});
        XTrain{chunki,1}=XTra(t:t+chunksize-1,:);
        t=t+chunksize;
    end
      
    %% Learn Bc
    alpha=OURparam.alpha;
    c=size(LTrain{1,1},2);
    nbits=OURparam.current_bits;

    B_c = sign(randn(nbits,c)); 
    B_c(B_c==0) = -1;
    
    Vector = Vector ./ sum(Vector.^2,2).^0.5;
    
    Vector = Vector';
    
    for i=1:OURparam.max_iter
        W=pinv(B_c*B_c'+alpha)*B_c*Vector';
        
        Q=W*Vector;
        
        for row=1:nbits
            B_c(row,:)=sign(Q(row,:)'-B_c(setdiff(1:nbits,row),:)'*W(setdiff(1:nbits,row),:)*W(row,:)')';        
        end
    end
    

    %% train
    for chunki = 1:OURparam.nchunks
        fprintf('-----chunk----- %3d\n', chunki);
        
        XTrain_new = XTrain{chunki,:};
        LTrain_new = LTrain{chunki,:};
        
        % Hash code learning
        if chunki == 1
            [BB,WW,PP,UU,PM,t] = train_IWMOH0(XTrain_new',LTrain_new',B_c,OURparam);
        else
            [BB,WW,PP,UU,PM,t] = train_IWMOH(XTrain_new',LTrain_new',BB,PP,PM,B_c,OURparam);
        end
        train_time_round(1,chunki) = t;
        fprintf('the %i chunk finished, train time is %d (s)\n',chunki,train_time_round(1,chunki));
        
        %% test
        fprintf('test beginning\n');
            
        E1=ones(1,OURparam.current_bits)*abs(UU{1,1}'*XQuery(:,1:OURparam.n_anchors)');
        E2=ones(1,OURparam.current_bits)*abs(UU{2,1}'*XQuery(:,OURparam.n_anchors+1:end)');
        
        E1(isnan(E1))=100;
        E2(isnan(E2))=100;
        
        PI1=1./(E1+0.0001);
        PI2=1./(E2+0.0001);
        P_and=PI1+PI2;
        PI1=PI1./P_and;
        PI2=PI2./P_and;
        
        PM1=log2(PM{1,1});
        PM2=log2(PM{2,1});
        PM_and=PM1+PM2;
        PM1=PM1/PM_and;
        PM2=PM2/PM_and;
        
        XQuery_B = compactbit((PI1.*(PM1*(WW{1,1}*XQuery(:,1:OURparam.n_anchors)'))+PI2.*(PM2*(WW{2,1}*XQuery(:,OURparam.n_anchors+1:end)')))'>0); 

        B = cell2mat(BB(1:chunki,:));
        XTrain_B = compactbit(B>0);
        
        %mAP
        DHamm = hammingDist(XQuery_B, XTrain_B);
        [~, orderH] = sort(DHamm, 2);
        eva(1,chunki) = mAP(orderH', cell2mat(LTrain(1:chunki,:)), LQuery);
        fprintf('the %i chunk : mAP=%d\n', chunki,eva(1,chunki));
    end
end

