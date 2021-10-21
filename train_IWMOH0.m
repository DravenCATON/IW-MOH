function [BB,WW,HH,UU,PM,t] = train_IWMOH0(XTrain_new,LTrain_new,B_c,param)
    
    % parameters
    theta = param.theta;
    delta = param.delta;
    
    n2 = size(LTrain_new,2);
    
    Xm1=XTrain_new(1:param.n_anchors,:);
    Xm2=XTrain_new(param.n_anchors+1:end,:); 
    

    tic
    % Step one
    B_new = sign(B_c*LTrain_new);
    
    HH{1,5} = B_new*Xm1';
    HH{1,6} = Xm1*Xm1';
    HH{1,7} = B_new*Xm2';
    HH{1,8} = Xm2*Xm2';
    
    BB{1,1} = B_new';
    
    % Step two
    WW{1,1}=HH{1,5}/(HH{1,6}+theta*eye(param.n_anchors));
    WW{2,1}=HH{1,7}/(HH{1,8}+theta*eye(param.n_anchors));
    
    HH{1,9} = Xm1*B_new';
    HH{1,10} = Xm2*B_new';
   
    % Step three
    UU{1,1}=(HH{1,6}+delta)\(HH{1,9}-HH{1,6}*WW{1,1}');
    UU{2,1}=(HH{1,8}+delta)\(HH{1,10}-HH{1,8}*WW{2,1}');
    
    PM{1,1}=n2*(1-param.current_a)+n2*param.current_a*param.current_b;
    PM{2,1}=n2*(1-param.current_a)+n2*param.current_a*(1-param.current_b);
    t=toc;
end

