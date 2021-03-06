function [net, info] = Fourlabel_last_decoder_AE_1to1_IVUS_L2_3B_LReLU_IpBN_withAug(imdb, inpt)

	% some common options
	% some common options
	trainer = @cnn_train_dag_ConvAE;

	opts.train.extractStatsFn = @extract_stats_segmentationAE;
	opts.train.batchSize = 20;
	opts.train.numEpochs = 120 ;
	opts.train.continue = true ;
	opts.train.gpus = [1] ;
	opts.train.learningRate = [1e-8*ones(1, 80),  1e-8*ones(1, 20), 1e-9*ones(1, 20)];
	%opts.train.learningRate = 1e-6;
    opts.train.weightDecay = 1e-3;
	opts.train.momentum = 0.9;
	opts.train.expDir = inpt;
	opts.train.savePlots = false;
	opts.train.numSubBatches = 5;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;
    
     opts.border = [8 8 8 8]; % tblr

    % augmenting data - Jitter and Fliplr
    augdata = zeros(size(imdb.images.data) + [sum(opts.border(1:2)) ...
    sum(opts.border(3:4)) 0 0], 'like', imdb.images.data);
    
    augdata(opts.border(1)+1:end-opts.border(2), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data;
    % Mirroring Borders for augdata
    augdata(1:opts.border(1), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data(opts.border(1):-1:1, ...
    :, :, :);
    augdata(end-opts.border(2)+1:end, ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data(end:-1:end-opts.border(2)+1, ...
    :, :, :);
    augdata(:, ...
    opts.border(3):-1:1, :, :) = augdata(:, ...
    opts.border(3)+1:2*opts.border(3), :, :);
    augdata(:, ...
    end-opts.border(4)+1:end, :, :) = augdata(:, ...
    end-opts.border(4):-1:end-2*opts.border(4)+1, :, :);


     % Augmenting Labels
    augLabels = zeros(size(imdb.images.labels) + [sum(opts.border(1:2)) ...
    sum(opts.border(3:4)) 0 0], 'like', imdb.images.labels);
    augLabels(opts.border(1)+1:end-opts.border(2), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.labels;
    % Mirroring Borders for augLabels
    augLabels(1:opts.border(1), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.labels(opts.border(1):-1:1, ...
    :, :, :);
    augLabels(end-opts.border(2)+1:end, ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.labels(end:-1:end-opts.border(2)+1, ...
    :, :, :);
    augLabels(:, ...
    opts.border(3):-1:1, :, :) = augLabels(:, ...
    opts.border(3)+1:2*opts.border(3), :, :);
    augLabels(:, ...
    end-opts.border(4)+1:end, :, :) = augLabels(:, ...
    end-opts.border(4):-1:end-2*opts.border(4)+1, :, :);


    

    imdb.images.augdata = augdata;
    imdb.images.augLabels = augLabels;
    clear augdata augLabels augData_M2 
    
    
	% organize data
	K = 2; % how many examples per domain	
	trainData = find(imdb.images.set == 1);
	valData = find(imdb.images.set == 2);
	
	% debuging code
	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];

	
	% network definition
	net = dagnn.DagNN() ;
    % Encoder Block 1 - M1
    %net.addLayer('bn0_M1', dagnn.BatchNorm('numChannels', 1), {'input_M1'}, {'bn0_M1'}, {'bn0f_M1', 'bn0b_M1', 'bn0m_M1'});
	net.addLayer('conv1_M1', dagnn.Conv('size', [7 7 1 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'input_M1'}, {'conv1_M1'},  {'conv1f_M1'  'conv1b_M1'});
	net.addLayer('bn1_M1', dagnn.BatchNorm('numChannels', 64), {'conv1_M1'}, {'bn1_M1'}, {'bn1f_M1', 'bn1b_M1', 'bn1m_M1'});
	net.addLayer('relu1_M1', dagnn.ReLU('leak', 0.01), {'bn1_M1'}, {'relu1_M1'}, {});
	net.addLayer('pool1_M1', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu1_M1'}, {'pool1_M1', 'pool1_indices_M1', 'sizes_pre_pool1_M1', 'sizes_post_pool1_M1'}, {});

    % Encoder Block 2 - M1
	net.addLayer('conv2_M1', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool1_M1'}, {'conv2_M1'},  {'conv2f_M1'  'conv2b_M1'});
	net.addLayer('bn2_M1', dagnn.BatchNorm('numChannels', 64), {'conv2_M1'}, {'bn2_M1'}, {'bn2f_M1', 'bn2b_M1', 'bn2m_M1'});
	net.addLayer('relu2_M1', dagnn.ReLU('leak', 0.01), {'bn2_M1'}, {'relu2_M1'}, {});
	net.addLayer('pool2_M1', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu2_M1'}, {'pool2_M1', 'pool2_indices_M1', 'sizes_pre_pool2_M1', 'sizes_post_pool2_M1'}, {});

    % Encoder Block 3 - M1
	net.addLayer('conv3_M1', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool2_M1'}, {'conv3_M1'},  {'conv3f_M1'  'conv3b_M1'});
	net.addLayer('bn3_M1', dagnn.BatchNorm('numChannels', 64), {'conv3_M1'}, {'bn3_M1'}, {'bn3f_M1', 'bn3b_M1', 'bn3m_M1'});
	net.addLayer('relu3_M1', dagnn.ReLU('leak', 0.01), {'bn3_M1'}, {'relu3_M1'}, {});
	net.addLayer('pool3_M1', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu3_M1'}, {'pool3_M1', 'pool3_indices_M1', 'sizes_pre_pool3_M1', 'sizes_post_pool3_M1'}, {});

    
    % Decoder Block 3 label1
    net.addLayer('unpool3x_M1_T1', dagnn.Unpooling(), {'pool3_M1', 'pool3_indices_M1', 'sizes_pre_pool3_M1', 'sizes_post_pool3_M1'}, {'unpool3x_M1_T1'}, {});
    net.addLayer('conv3x_T1', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool3x_M1_T1'}, {'conv3x_T1'},  {'conv3xf_T1'  'conv3xb_T1'});
    net.addLayer('bn3x_T1', dagnn.BatchNorm('numChannels', 64), {'conv3x_T1'}, {'bn3x_T1'}, {'bn3xf_T1', 'bn3xb_T1', 'bn3xm_T1'});
    net.addLayer('relu3x_T1', dagnn.ReLU('leak', 0.01), {'bn3x_T1'}, {'relu3x_T1'}, {});
    
    % Decoder Block 3 label2
    net.addLayer('unpool3x_M1_T2', dagnn.Unpooling(), {'pool3_M1', 'pool3_indices_M1', 'sizes_pre_pool3_M1', 'sizes_post_pool3_M1'}, {'unpool3x_M1_T2'}, {});
    net.addLayer('conv3x_T2', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool3x_M1_T2'}, {'conv3x_T2'},  {'conv3xf_T2'  'conv3xb_T2'});
    net.addLayer('bn3x_T2', dagnn.BatchNorm('numChannels', 64), {'conv3x_T2'}, {'bn3x_T2'}, {'bn3xf_T2', 'bn3xb_T2', 'bn3xm_T2'});
    net.addLayer('relu3x_T2', dagnn.ReLU('leak', 0.01), {'bn3x_T2'}, {'relu3x_T2'}, {});

    % Decoder Block 3 label3
    net.addLayer('unpool3x_M1_T3', dagnn.Unpooling(), {'pool3_M1', 'pool3_indices_M1', 'sizes_pre_pool3_M1', 'sizes_post_pool3_M1'}, {'unpool3x_M1_T3'}, {});
    net.addLayer('conv3x_T3', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool3x_M1_T3'}, {'conv3x_T3'},  {'conv3xf_T3'  'conv3xb_T3'});
    net.addLayer('bn3x_T3', dagnn.BatchNorm('numChannels', 64), {'conv3x_T3'}, {'bn3x_T3'}, {'bn3xf_T3', 'bn3xb_T3', 'bn3xm_T3'});
    net.addLayer('relu3x_T3', dagnn.ReLU('leak', 0.01), {'bn3x_T3'}, {'relu3x_T3'}, {});

    % Decoder Block 3 label4
    net.addLayer('unpool3x_M1_T4', dagnn.Unpooling(), {'pool3_M1', 'pool3_indices_M1', 'sizes_pre_pool3_M1', 'sizes_post_pool3_M1'}, {'unpool3x_M1_T4'}, {});
    net.addLayer('conv3x_T4', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool3x_M1_T4'}, {'conv3x_T4'},  {'conv3xf_T4'  'conv3xb_T4'});
    net.addLayer('bn3x_T4', dagnn.BatchNorm('numChannels', 64), {'conv3x_T4'}, {'bn3x_T4'}, {'bn3xf_T4', 'bn3xb_T4', 'bn3xm_T4'});
    net.addLayer('relu3x_T4', dagnn.ReLU('leak', 0.01), {'bn3x_T4'}, {'relu3x_T4'}, {});

    % Decoder Block 3 label4
    net.addLayer('unpool3x_M1_T5', dagnn.Unpooling(), {'pool3_M1', 'pool3_indices_M1', 'sizes_pre_pool3_M1', 'sizes_post_pool3_M1'}, {'unpool3x_M1_T5'}, {});
    net.addLayer('conv3x_T5', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'unpool3x_M1_T5'}, {'conv3x_T5'},  {'conv3xf_T5'  'conv3xb_T5'});
    net.addLayer('bn3x_T5', dagnn.BatchNorm('numChannels', 64), {'conv3x_T5'}, {'bn3x_T5'}, {'bn3xf_T5', 'bn3xb_T5', 'bn3xm_T5'});
    net.addLayer('relu3x_T5', dagnn.ReLU('leak', 0.01), {'bn3x_T5'}, {'relu3x_T5'}, {});


    
    % Decoder Block 2 label1
    net.addLayer('unpool2x_M1_T1', dagnn.Unpooling(), {'relu3x_T1', 'pool2_indices_M1', 'sizes_pre_pool2_M1', 'sizes_post_pool2_M1'}, {'unpool2x_M1_T1'}, {});
    net.addLayer('concat2x_T1', dagnn.Concat('dim',3), {'unpool2x_M1_T1','relu2_M1_T1'}, {'concat2x_T1'});
	net.addLayer('dimReduce2x_T1', dagnn.Conv('size', [1 1 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'concat2x_T1'}, {'dimReduce2x_T1'},  {'conv_dimReduce_2xf_T1'  'conv_dimReduce_2xb_T1'});
	net.addLayer('conv2x_T1', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'dimReduce2x_T1'}, {'conv2x_T1'},  {'conv2xf_T1'  'conv2xb_T1'});
	net.addLayer('bn2x_T1', dagnn.BatchNorm('numChannels', 64), {'conv2x_T1'}, {'bn2x_T1'}, {'bn2xf_T1', 'bn2xb_T1', 'bn2xm_T1'});
	net.addLayer('relu2x_T1', dagnn.ReLU('leak', 0.01), {'bn2x_T1'}, {'relu2x_T1'}, {});
%     net.addLayer('drop', dagnn.DropOut('rate', 0.5), {'relu7'}, {'drop'}, {});

 % Decoder Block 2 label2
    net.addLayer('unpool2x_M1_T2', dagnn.Unpooling(), {'relu3x_T2', 'pool2_indices_M1', 'sizes_pre_pool2_M1', 'sizes_post_pool2_M1'}, {'unpool2x_M1_T2'}, {});
    net.addLayer('concat2x_T2', dagnn.Concat('dim',3), {'unpool2x_M1_T2','relu2_M1_T2'}, {'concat2x_T2'});
    net.addLayer('dimReduce2x_T2', dagnn.Conv('size', [1 1 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'concat2x_T2'}, {'dimReduce2x_T2'},  {'conv_dimReduce_2xf_T2'  'conv_dimReduce_2xb_T2'});
    net.addLayer('conv2x_T2', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'dimReduce2x_T2'}, {'conv2x_T2'},  {'conv2xf_T2'  'conv2xb_T2'});
    net.addLayer('bn2x_T2', dagnn.BatchNorm('numChannels', 64), {'conv2x_T2'}, {'bn2x_T2'}, {'bn2xf_T2', 'bn2xb_T2', 'bn2xm_T2'});
    net.addLayer('relu2x_T2', dagnn.ReLU('leak', 0.01), {'bn2x_T2'}, {'relu2x_T2'}, {});

 % Decoder Block 2 label3
    net.addLayer('unpool2x_M1_T3', dagnn.Unpooling(), {'relu3x_T3', 'pool2_indices_M1', 'sizes_pre_pool2_M1', 'sizes_post_pool2_M1'}, {'unpool2x_M1_T3'}, {});
    net.addLayer('concat2x_T3', dagnn.Concat('dim',3), {'unpool2x_M1_T3','relu2_M1_T3'}, {'concat2x_T3'});
    net.addLayer('dimReduce2x_T3', dagnn.Conv('size', [1 1 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'concat2x_T3'}, {'dimReduce2x_T3'},  {'conv_dimReduce_2xf_T3'  'conv_dimReduce_2xb_T3'});
    net.addLayer('conv2x_T3', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'dimReduce2x_T3'}, {'conv2x_T3'},  {'conv2xf_T3'  'conv2xb_T3'});
    net.addLayer('bn2x_T3', dagnn.BatchNorm('numChannels', 64), {'conv2x_T3'}, {'bn2x_T3'}, {'bn2xf_T3', 'bn2xb_T3', 'bn2xm_T3'});
    net.addLayer('relu2x_T3', dagnn.ReLU('leak', 0.01), {'bn2x_T3'}, {'relu2x_T3'}, {});

 % Decoder Block 2 label4
    net.addLayer('unpool2x_M1_T4', dagnn.Unpooling(), {'relu3x_T4', 'pool2_indices_M1', 'sizes_pre_pool2_M1', 'sizes_post_pool2_M1'}, {'unpool2x_M1_T4'}, {});
    net.addLayer('concat2x_T4', dagnn.Concat('dim',3), {'unpool2x_M1_T4','relu2_M1_T4'}, {'concat2x_T4'});
    net.addLayer('dimReduce2x_T4', dagnn.Conv('size', [1 1 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'concat2x_T4'}, {'dimReduce2x_T4'},  {'conv_dimReduce_2xf_T4'  'conv_dimReduce_2xb_T4'});
    net.addLayer('conv2x_T4', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'dimReduce2x_T4'}, {'conv2x_T4'},  {'conv2xf_T4'  'conv2xb_T4'});
    net.addLayer('bn2x_T4', dagnn.BatchNorm('numChannels', 64), {'conv2x_T4'}, {'bn2x_T4'}, {'bn2xf_T4', 'bn2xb_T4', 'bn2xm_T4'});
    net.addLayer('relu2x_T4', dagnn.ReLU('leak', 0.01), {'bn2x_T4'}, {'relu2x_T4'}, {});

 % Decoder Block 2 label5
    net.addLayer('unpool2x_M1_T5', dagnn.Unpooling(), {'relu3x_T5', 'pool2_indices_M1', 'sizes_pre_pool2_M1', 'sizes_post_pool2_M1'}, {'unpool2x_M1_T5'}, {});
    net.addLayer('concat2x_T5', dagnn.Concat('dim',3), {'unpool2x_M1_T5','relu2_M1_T5'}, {'concat2x_T5'});
    net.addLayer('dimReduce2x_T5', dagnn.Conv('size', [1 1 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'concat2x_T5'}, {'dimReduce2x_T5'},  {'conv_dimReduce_2xf_T5'  'conv_dimReduce_2xb_T5'});
    net.addLayer('conv2x_T5', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'dimReduce2x_T5'}, {'conv2x_T5'},  {'conv2xf_T5'  'conv2xb_T5'});
    net.addLayer('bn2x_T5', dagnn.BatchNorm('numChannels', 64), {'conv2x_T5'}, {'bn2x_T5'}, {'bn2xf_T5', 'bn2xb_T5', 'bn2xm_T5'});
    net.addLayer('relu2x_T5', dagnn.ReLU('leak', 0.01), {'bn2x_T5'}, {'relu2x_T5'}, {});




    % Decoder Block 1 label1
    net.addLayer('unpool1x_M1_T1', dagnn.Unpooling(), {'relu2x_T1', 'pool1_indices_M1', 'sizes_pre_pool1_M1', 'sizes_post_pool1_M1'}, {'unpool1x_M1_T1'}, {});
    net.addLayer('concat1x_T1', dagnn.Concat('dim',3), {'unpool1x_M1_T1','relu1_M1'}, {'concat1x_T1'});
	net.addLayer('dimReduce1x_T1', dagnn.Conv('size', [1 1 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'concat1x_T1'}, {'dimReduce1x_T1'},  {'conv_dimReduce_1xf_T1'  'conv_dimReduce_1xb_T1'});
	net.addLayer('conv1x_T1', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'dimReduce1x_T1'}, {'conv1x_T1'},  {'conv1xf_T1'  'conv1xb_T1'});
	net.addLayer('bn1x_T1', dagnn.BatchNorm('numChannels', 64), {'conv1x_T1'}, {'bn1x_T1'}, {'bn1xf_T1', 'bn1xb_T1', 'bn1xm_T1'});
	net.addLayer('relu1x_T1', dagnn.ReLU('leak', 0.01), {'bn1x_T1'}, {'relu1x_T1'}, {});
    
    net.addLayer('drop_T1', dagnn.DropOut('rate', 0.5), {'relu1x_T1'}, {'drop_T1'}, {});
     
    net.addLayer('reconstruction_T1', dagnn.Conv('size', [1 1 64 1], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop_T1'}, {'reconstruction_T1'},  {'classf_T1'  'classb_T1'});
	net.addLayer('sigmoid1_T1', dagnn.Sigmoid, {'reconstruction_T1'}, {'sigmoid1_T1'}, {});
    %net.addLayer('objective_T1', dagnn.LossAE('weights', 'true'), {'sigmoid1_T1','label1'}, {'objective_T1'});
    
    
    % Decoder Block 1 label2
    net.addLayer('unpool1x_M1_T2', dagnn.Unpooling(), {'relu2x_T2', 'pool1_indices_M1', 'sizes_pre_pool1_M1', 'sizes_post_pool1_M1'}, {'unpool1x_M1_T2'}, {});
    net.addLayer('concat1x_T2', dagnn.Concat('dim',3), {'unpool1x_M1_T2','relu1_M1'}, {'concat1x_T2'});
	net.addLayer('dimReduce1x_T2', dagnn.Conv('size', [1 1 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'concat1x_T2'}, {'dimReduce1x_T2'},  {'conv_dimReduce_1xf_T2'  'conv_dimReduce_1xb_T2'});
	net.addLayer('conv1x_T2', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'dimReduce1x_T2'}, {'conv1x_T2'},  {'conv1xf_T2'  'conv1xb_T2'});
	net.addLayer('bn1x_T2', dagnn.BatchNorm('numChannels', 64), {'conv1x_T2'}, {'bn1x_T2'}, {'bn1xf_T2', 'bn1xb_T2', 'bn1xm_T2'});
	net.addLayer('relu1x_T2', dagnn.ReLU('leak', 0.01), {'bn1x_T2'}, {'relu1x_T2'}, {});
    
    net.addLayer('drop_T2', dagnn.DropOut('rate', 0.5), {'relu1x_T2'}, {'drop_T2'}, {});
     
    net.addLayer('reconstruction_T2', dagnn.Conv('size', [1 1 64 1], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop_T2'}, {'reconstruction_T2'},  {'classf_T2'  'classb_T2'});
	net.addLayer('sigmoid1_T2', dagnn.Sigmoid, {'reconstruction_T2'}, {'sigmoid1_T2'}, {});
    %net.addLayer('objective_T2', dagnn.LossAE('weights', 'true'), {'sigmoid1_T2','label2'}, {'objective_T2'});
    
    
    % Decoder Block 1 label3
    net.addLayer('unpool1x_M1_T3', dagnn.Unpooling(), {'relu2x_T3', 'pool1_indices_M1', 'sizes_pre_pool1_M1', 'sizes_post_pool1_M1'}, {'unpool1x_M1_T3'}, {});
    net.addLayer('concat1x_T3', dagnn.Concat('dim',3), {'unpool1x_M1_T3','relu1_M1'}, {'concat1x_T3'});
	net.addLayer('dimReduce1x_T3', dagnn.Conv('size', [1 1 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'concat1x_T3'}, {'dimReduce1x_T3'},  {'conv_dimReduce_1xf_T3'  'conv_dimReduce_1xb_T3'});
	net.addLayer('conv1x_T3', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'dimReduce1x_T3'}, {'conv1x_T3'},  {'conv1xf_T3'  'conv1xb_T3'});
	net.addLayer('bn1x_T3', dagnn.BatchNorm('numChannels', 64), {'conv1x_T3'}, {'bn1x_T3'}, {'bn1xf_T3', 'bn1xb_T3', 'bn1xm_T3'});
	net.addLayer('relu1x_T3', dagnn.ReLU('leak', 0.01), {'bn1x_T3'}, {'relu1x_T3'}, {});
    
    net.addLayer('drop_T3', dagnn.DropOut('rate', 0.5), {'relu1x_T3'}, {'drop_T3'}, {});
     
    net.addLayer('reconstruction_T3', dagnn.Conv('size', [1 1 64 1], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop_T3'}, {'reconstruction_T3'},  {'classf_T3'  'classb_T3'});
	net.addLayer('sigmoid1_T3', dagnn.Sigmoid, {'reconstruction_T3'}, {'sigmoid1_T3'}, {});
    %net.addLayer('objective_T3', dagnn.LossAE('weights', 'true'), {'sigmoid1_T3','label3'}, {'objective_T3'});
    
    
    
    % Decoder Block 1 label4
    net.addLayer('unpool1x_M1_T4', dagnn.Unpooling(), {'relu2x_T4', 'pool1_indices_M1', 'sizes_pre_pool1_M1', 'sizes_post_pool1_M1'}, {'unpool1x_M1_T4'}, {});
    net.addLayer('concat1x_T4', dagnn.Concat('dim',3), {'unpool1x_M1_T4','relu1_M1'}, {'concat1x_T4'});
	net.addLayer('dimReduce1x_T4', dagnn.Conv('size', [1 1 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'concat1x_T4'}, {'dimReduce1x_T4'},  {'conv_dimReduce_1xf_T4'  'conv_dimReduce_1xb_T4'});
	net.addLayer('conv1x_T4', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'dimReduce1x_T4'}, {'conv1x_T4'},  {'conv1xf_T4'  'conv1xb_T4'});
	net.addLayer('bn1x_T4', dagnn.BatchNorm('numChannels', 64), {'conv1x_T4'}, {'bn1x_T4'}, {'bn1xf_T4', 'bn1xb_T4', 'bn1xm_T4'});
	net.addLayer('relu1x_T4', dagnn.ReLU('leak', 0.01), {'bn1x_T4'}, {'relu1x_T4'}, {});
    
    net.addLayer('drop_T4', dagnn.DropOut('rate', 0.5), {'relu1x_T4'}, {'drop_T4'}, {});
     
    net.addLayer('reconstruction_T4', dagnn.Conv('size', [1 1 64 1], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop_T4'}, {'reconstruction_T4'},  {'classf_T4'  'classb_T4'});
	net.addLayer('sigmoid1_T4', dagnn.Sigmoid, {'reconstruction_T4'}, {'sigmoid1_T4'}, {});
    %net.addLayer('objective_T4', dagnn.LossAE('weights', 'true'), {'sigmoid1_T4','label4'}, {'objective_T4'});

% Decoder Block 1 label5
    net.addLayer('unpool1x_M1_T5', dagnn.Unpooling(), {'relu2x_T5', 'pool1_indices_M1', 'sizes_pre_pool1_M1', 'sizes_post_pool1_M1'}, {'unpool1x_M1_T5'}, {});
    net.addLayer('concat1x_T5', dagnn.Concat('dim',3), {'unpool1x_M1_T5','relu1_M1'}, {'concat1x_T5'});
	net.addLayer('dimReduce1x_T5', dagnn.Conv('size', [1 1 128 64], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'concat1x_T5'}, {'dimReduce1x_T5'},  {'conv_dimReduce_1xf_T5'  'conv_dimReduce_1xb_T5'});
	net.addLayer('conv1x_T5', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'dimReduce1x_T5'}, {'conv1x_T5'},  {'conv1xf_T5'  'conv1xb_T5'});
	net.addLayer('bn1x_T5', dagnn.BatchNorm('numChannels', 64), {'conv1x_T5'}, {'bn1x_T5'}, {'bn1xf_T5', 'bn1xb_T5', 'bn1xm_T5'});
	net.addLayer('relu1x_T5', dagnn.ReLU('leak', 0.01), {'bn1x_T5'}, {'relu1x_T5'}, {});
    
    net.addLayer('drop_T5', dagnn.DropOut('rate', 0.5), {'relu1x_T5'}, {'drop_T5'}, {});
     
    net.addLayer('reconstruction_T5', dagnn.Conv('size', [1 1 64 1], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'drop_T5'}, {'reconstruction_T5'},  {'classf_T5'  'classb_T5'});
	net.addLayer('sigmoid1_T5', dagnn.Sigmoid, {'reconstruction_T5'}, {'sigmoid1_T5'}, {});
    %net.addLayer('objective_T5', dagnn.LossAE('weights', 'true'), {'sigmoid1_T5','label5'}, {'objective_T5'});
    
    net.addLayer('sigmoids_concat',dagnn.Concat('dim',3),{'sigmoid1_T1','sigmoid1_T2','sigmoid1_T3','sigmoid1_T4','sigmoid1_T5'},{'sigmoids_concat'});
    net.addLayer('objective', dagnn.LossAE('weights', 'true'), {'sigmoids_concat','labels'}, {'objective'});
    
    
    
    % -- end of the network
    
	% do the training!
    initNet(net); %Xavier Initialization
	
	net.conserveMemory = false;

	info = trainer(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'train', trainData, 'val', valData) ;
end


% function on charge of creating a batch of images + labels
function inputs = getBatch(opts, imdb, batch)
if imdb.images.set(batch(1))==1  % training
  numAug = 2;
  images_M1 = imdb.images.data(:,:,:,batch); 
  labels = imdb.images.labels(:,:,:,batch); 
  
  for idx = 1:numAug
  sz0 = size(imdb.images.augdata);
  sz = size(imdb.images.data);
  loc = [randi(sz0(1)-sz(1)+1) randi(sz0(2)-sz(2)+1)];
  images_M1_Current = imdb.images.augdata(loc(1):loc(1)+sz(1)-1, ...
    loc(2):loc(2)+sz(2)-1, :, batch); 
  labels_Current = imdb.images.augLabels(loc(1):loc(1)+sz(1)-1, ...
    loc(2):loc(2)+sz(2)-1, :, batch); 
      % flip data randomly
    if rand > 0.5, images_M1_Current = fliplr(images_M1_Current) ;  labels_Current = fliplr(labels_Current); end

  images_M1 = cat(4,images_M1,images_M1_Current);
  labels = cat(4,labels,labels_Current);
  end
else                              % validating / testing
  images_M1 = imdb.images.data(:,:,:,batch); 
  labels = imdb.images.labels(:,:,:,batch); 
end
if opts.useGpu > 0
    images_M1 = gpuArray(images_M1);
    labels = gpuArray(labels); 
end

% label1=labels(:,:,1,:);
% label2=labels(:,:,2,:);
% label3=labels(:,:,3,:);
% label4=labels(:,:,4,:);
% label5=labels(:,:,5,:);

    inputs = {'input_M1', images_M1, 'labels', labels};