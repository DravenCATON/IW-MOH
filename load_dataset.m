function [train_param,XTrain,LTrain,XQuery,LQuery,Vector] = load_dataset(train_param)
    fprintf(['-------load dataset------', '\n']);
    load([train_param.ds_name,'_deep.mat']);
    load([train_param.ds_name,'_Groundtrue_Vec.mat']);

    if strcmp(train_param.ds_name, 'MIRFLICKR')
        train_param.chunksize = 2000;

        train_param.image_feature_size=4096;
        train_param.text_feature_size=1386;
        
        X = [X Y];

        R = randperm(size(L,1));
        queryInds = R(1:2000);
        sampleInds = R(2001:end);

        train_param.nchunks = ceil(length(sampleInds)/train_param.chunksize);

        XTrain = cell(train_param.nchunks,1);
        LTrain = cell(train_param.nchunks,1);
        for subi = 1:train_param.nchunks-1
            XTrain{subi,1} = X(sampleInds(train_param.chunksize*(subi-1)+1:train_param.chunksize*subi),:);

            if train_param.train_unpair
                %remove text feature
                XTrain{subi,1}(train_param.chunksize*(1-train_param.current_a)+1:...
                                      train_param.chunksize*(1-train_param.current_a)+...
                                      train_param.chunksize*train_param.current_a*train_param.current_b,...
                                      train_param.image_feature_size+1:end)=0;
                %remove image feature                      
                XTrain{subi,1}(train_param.chunksize*(1-train_param.current_a)+...
                                      train_param.chunksize*train_param.current_a*train_param.current_b+1:...
                                      end,...
                                      1:train_param.image_feature_size)=0;
            end

            LTrain{subi,1} = L(sampleInds(train_param.chunksize*(subi-1)+1:train_param.chunksize*subi),:);
        end
        XTrain{train_param.nchunks,1} = X(sampleInds(train_param.chunksize*subi+1:end),:);

        if train_param.train_unpair
            %remove text feature
            XTrain{train_param.nchunks,1}(floor(size(XTrain{train_param.nchunks,1},1)*(1-train_param.current_a))+1:...
                                          floor(size(XTrain{train_param.nchunks,1},1)*(1-train_param.current_a))+...
                                          floor(ceil(size(XTrain{train_param.nchunks,1},1)*train_param.current_a)*train_param.current_b),...
                                          train_param.image_feature_size+1:end)=0;
            %remove image feature                          
            XTrain{train_param.nchunks,1}(floor(size(XTrain{train_param.nchunks,1},1)*(1-train_param.current_a))+...
                                          floor(ceil(size(XTrain{train_param.nchunks,1},1)*train_param.current_a)*train_param.current_b)+1:...
                                          end,...
                                          1:train_param.image_feature_size)=0;
        end

        LTrain{train_param.nchunks,1} = L(sampleInds(train_param.chunksize*subi+1:end),:);

        XQuery = X(queryInds, :); LQuery = L(queryInds, :);
        if train_param.query_unpair
            XQuery(2000*(1-train_param.current_a)+1:2000*(1-train_param.current_a)+...
                    2000*train_param.current_a*train_param.current_b,train_param.image_feature_size+1:end)=0;
            XQuery(2000*(1-train_param.current_a)+2000*train_param.current_a*train_param.current_b+1:end,...
                    1:train_param.image_feature_size)=0;
        end
        
        Vector=double(mir_gt_vec);

        clear X L subi queryInds sampleInds R Image Tag Label

    elseif strcmp(train_param.ds_name, 'NUSWIDE21')
        train_param.chunksize = 10000;

        train_param.image_feature_size=4096;
        train_param.text_feature_size=5018;
        
        I_tr=double(I_tr);
        I_te=double(I_te);
        X = [I_tr T_tr; I_te T_te];L = [L_tr; L_te];

        R = randperm(size(L,1));
        queryInds = R(1:2000);
        sampleInds = R(2001:end);

        train_param.nchunks = ceil(length(sampleInds)/train_param.chunksize);

        XTrain = cell(train_param.nchunks,1);
        LTrain = cell(train_param.nchunks,1);
        for subi = 1:train_param.nchunks-1
            XTrain{subi,1} = X(sampleInds(train_param.chunksize*(subi-1)+1:train_param.chunksize*subi),:);
            if train_param.train_unpair
                %remove text feature
                XTrain{subi,1}(train_param.chunksize*(1-train_param.current_a)+1:...
                                          train_param.chunksize*(1-train_param.current_a)+...
                                          train_param.chunksize*train_param.current_a*train_param.current_b,...
                                          train_param.image_feature_size+1:end)=0;
                %remove image feature                      
                XTrain{subi,1}(train_param.chunksize*(1-train_param.current_a)+...
                                          train_param.chunksize*train_param.current_a*train_param.current_b+1:...
                                          end,...
                                          1:train_param.image_feature_size)=0;
            end

            LTrain{subi,1} = L(sampleInds(train_param.chunksize*(subi-1)+1:train_param.chunksize*subi),:);
        end
        XTrain{train_param.nchunks,1} = X(sampleInds(train_param.chunksize*subi+1:end),:);
        if train_param.train_unpair
            %remove text feature
            XTrain{train_param.nchunks,1}(floor(size(XTrain{train_param.nchunks,1},1)*(1-train_param.current_a))+1:...
                                          floor(size(XTrain{train_param.nchunks,1},1)*(1-train_param.current_a))+...
                                          floor(ceil(size(XTrain{train_param.nchunks,1},1)*train_param.current_a)*train_param.current_b),...
                                          train_param.image_feature_size+1:end)=0;
            %remove image feature                          
            XTrain{train_param.nchunks,1}(floor(size(XTrain{train_param.nchunks,1},1)*(1-train_param.current_a))+...
                                          floor(ceil(size(XTrain{train_param.nchunks,1},1)*train_param.current_a)*train_param.current_b)+1:...
                                          end,...
                                          1:train_param.image_feature_size)=0;
        end

        LTrain{train_param.nchunks,1} = L(sampleInds(train_param.chunksize*subi+1:end),:);

        XQuery = X(queryInds, :); LQuery = L(queryInds, :);
        if train_param.query_unpair
            XQuery(2000*(1-train_param.current_a)+1:2000*(1-train_param.current_a)+...
                    2000*train_param.current_a*train_param.current_b,train_param.image_feature_size+1:end)=0;
            XQuery(2000*(1-train_param.current_a)+2000*train_param.current_a*train_param.current_b+1:end,...
                    1:train_param.image_feature_size)=0;
        end    
        
        Vector=double(nus_gt_vec);
        
        clear X L subi queryInds sampleInds R
    end
    fprintf('-------load data finished-------\n');
    clear I_tr I_te L_tr L_te T_tr T_te
end

