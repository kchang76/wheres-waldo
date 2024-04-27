function utils = data_utils()
    utils = {};
    utils.augment = @augment;
    utils.flatten_ds = @flatten_ds;
    utils.calculate_anchors = @calculate_anchors;
    utils.train_test_split = @train_test_split;
    utils.train_validation_test_split = @train_validation_test_split;
    utils.get_metrics_report = @get_metrics_report;
    utils.chunk_image = @chunk_image;
end

% Accepts image ds/bounding label data store and performs data augmentation
function out = augment(data)
    
    % TO DO: need to implement
    %bboxcrop - Crop images for bounding box
    %bboxresize - Resize bounding boxes by fixed scale

    % Unpack original data.
    I = data{1};
    boxes = data{2};
    labels = data{3};
    
    %jitterColorHSV - Randomly adjust image hue, saturation, brightness, or contrast of color images
    I = jitterColorHSV(I,"Brightness",0.3,"Contrast",0.4,"Saturation",0.2);

    %imnoise - Add random Gaussian, Poisson, salt and pepper, or multiplicative noise
    I = imnoise(I,"gaussian");
    
    %Define random affine transform.
    tform = randomAffine2d("XReflection",true,'Rotation',[-30 30]);
    rout = affineOutputView(size(I),tform);
    
    %bboxwarp - Apply random reflection, rotation, scale, shear, and translation to images
    augmentedImage = imwarp(I,tform,"OutputView",rout);
    [augmentedBoxes, valid] = bboxwarp(boxes,tform,rout,'OverlapThreshold',0.5);
    augmentedLabels = labels(valid);
    
    % Return augmented data.
    out = {augmentedImage,augmentedBoxes,augmentedLabels};
end


function flat_ds = flatten_ds(ds_collection)
    image_ds = ds_collection{1}.UnderlyingDatastores{1};
    bl_ds = ds_collection{1}.UnderlyingDatastores{2};

    for i=2:length(ds_collection)
        new_img_ds = ds_collection{i}.UnderlyingDatastores{1};
        new_bl_ds = ds_collection{1}.UnderlyingDatastores{2};
    end

    % TO DO finish function
    flat_ds = None;
end


% Calculates anchors for NN based on training data
% Importance of Anchor boxes is typical ratios of boundary boxes for
%   network to estimate; aids detection process. Output meant for network
%   with 2 output nodes.
function anchors = calculate_anchors(training_ds, numAnchors)    
     if ~exist('numAnchors','var')
          numAnchors = 4;
     end

    [anchors, ~] = estimateAnchorBoxes(training_ds,numAnchors);
    area = anchors(:,1).*anchors(:,2);
    [~,idx] = sort(area,"descend");
    anchors = anchors(idx,:);

    % Split anchors for 2 output nodes
    anchors = {anchors(1:floor(length(anchors)/2),:); ...
        anchors(floor(length(anchors)/2)+1:end,:)};
end

% Shuffles and splits datastore into training and test data
%   Defaults to 80% training, 20% test
function [train_ds, test_ds] = train_test_split(img_ds, bl_ds, train_split)
    if ~exist('train_split','var')
          train_split = 0.8;
    end

    rng("default");
    % shuffle images in datastores
    combined_ds = combine(img_ds,bl_ds);
    combined_ds = shuffle(combined_ds);
    image_datastore = combined_ds.UnderlyingDatastores{1};
    box_label_datastore = combined_ds.UnderlyingDatastores{2};
    
    split = floor( train_split * length(image_datastore.Files) );
    
    % First split of data is training
    train_img_ds = subset(image_datastore, 1:split);
    train_bl_ds = subset(box_label_datastore, 1:split);
    
    % Last split of data is test
    test_img_ds = subset(image_datastore, split+1:length(image_datastore.Files));
    test_bl_ds = subset(box_label_datastore, split+1:length(image_datastore.Files));

    % Combine for output
    train_ds = combine(train_img_ds, train_bl_ds);
    test_ds = combine(test_img_ds, test_bl_ds);
end

% Shuffles and splits datastore into training, validation, and test data
%   Defaults to 60% training, 20% validation, 20% test
function [train_ds, validation_ds, test_ds] = train_validation_test_split(img_ds, bl_ds, train_split)
    if ~exist('train_split','var')
          train_split = 0.6;
    end

    rng("default");
    % shuffle images in datastores
    combined_ds = combine(img_ds,bl_ds);
    combined_ds = shuffle(combined_ds);
    image_datastore = combined_ds.UnderlyingDatastores{1};
    box_label_datastore = combined_ds.UnderlyingDatastores{2};
    
    split = floor( train_split * length(image_datastore.Files) );
    
    % First split of data is training
    train_img_ds = subset(image_datastore, 1:split);
    train_bl_ds = subset(box_label_datastore, 1:split);

    validation_split = floor( mean([1.0, train_split]) * length(image_datastore.Files) );
    % Second split of data is validation
    valid_img_ds = subset(image_datastore, split+1:validation_split);
    valid_bl_ds = subset(box_label_datastore, split+1:validation_split);

    % Last split of data is test
    test_img_ds = subset(image_datastore, validation_split+1:length(image_datastore.Files));
    test_bl_ds = subset(box_label_datastore, validation_split+1:length(image_datastore.Files));

    % Combine for output
    train_ds = combine(train_img_ds, train_bl_ds);
    validation_ds = combine(valid_img_ds, valid_bl_ds);
    test_ds = combine(test_img_ds, test_bl_ds);
end

function [] = get_metrics_report (detector, test_ds, class_name, debug)
    if ~exist('debug','var')
          debug = false;
    end

    % get test metrics on different thresholds
    for threshold=[0.01, 0.1, 0.2, 0.5, 0.7, 0.9]
        detectionResults = detect(detector,test_ds,Threshold=threshold);
        % Require at least 33% overlap in bounding boxes for label to be
        % counted
        metrics = evaluateObjectDetection(detectionResults,test_ds,0.33);
        fprintf("Confidence Threshold:\t %f\n", threshold);

        if (length(metrics.ClassMetrics.Precision{class_name}) > 1)
            fprintf("  Precision:\t%f\tRecall:\t%f\n", ...
                metrics.ClassMetrics.Precision{class_name}(2), ...
                metrics.ClassMetrics.Recall{class_name}(2));
        else 
            fprintf("  Precision:\t%f\tRecall:\t%f\n", ...
                metrics.ClassMetrics.Precision{class_name}(1), ...
                metrics.ClassMetrics.Recall{class_name}(1));
        end
    
        % debug logic to show drawn boxes on test cases
        if debug
            im = imread(test_ds.UnderlyingDatastores{1}.Files{2});
            im = insertObjectAnnotation(im,"rectangle", ...
                detectionResults.Boxes{2}, detectionResults.Scores{2});
            figure('Name',"Threshold " + threshold);
            imshow(im);
        end
    end
end

function [] = chunk_image(filepath, source, chunk_size)
    if ~exist('chunk_size','var')
          chunk_size = 128;
    end
    mkdir(sprintf("dataset/%i/%s", chunk_size,source));

    [~,filename,~] = fileparts(filepath);
    im = imread(filepath);
    im_size = size(im);

    % Decrease resolution if image is too large
    % if (im_size(1) > 1400)
    %     targetSize = [1400 NaN];
    %     im = imresize(im,targetSize);
    %     im_size = size(im);
    % end
    % if (im_size(2) > 2200)
    %     targetSize = [NaN 2200];
    %     im = imresize(im,targetSize);
    %     im_size = size(im);
    % end

    % Save image as new subsections of chunk_size by chunk_size
    for row_index=1:ceil(im_size(1)/chunk_size)
        r_end = min((row_index)*chunk_size, im_size(1));
        r_start = r_end - chunk_size + 1;

        for col_index=1:ceil(im_size(2)/chunk_size)
            c_end = min((col_index)*chunk_size, im_size(2));
            c_start = c_end - chunk_size + 1;

            im_subsection = im(r_start:r_end, c_start:c_end, :);
            imwrite(im_subsection, sprintf("dataset/%i/%s/%s_%i_%i.jpg", chunk_size,source,filename,row_index,col_index));
        end
    end
end

