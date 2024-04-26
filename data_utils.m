function utils = data_utils()
    utils = {};
    utils.augment = @augment;
    utils.calculate_anchors = @calculate_anchors;
    utils.train_test_split = @train_test_split;
    utils.train_validation_test_split = @train_validation_test_split;
    utils.get_metrics_report = @get_metrics_report;
end

% Accepts image ds/bounding label data store and performs data augmentation
function augmented_ds = augment(img_ds, bl_ds)
    
    % TO DO: need to implement

    %bboxcrop - Crop images for bounding box
    %bboxresize - Resize bounding boxes by fixed scale
    %imnoise - Add random Gaussian, Poisson, salt and pepper, or multiplicative noise
    %jitterColorHSV - Randomly adjust image hue, saturation, brightness, or contrast of color images
    %bboxwarp - Apply random reflection, rotation, scale, shear, and translation to images
    augmented_ds = None;
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
    for threshold=[0.1, 0.2, 0.5, 0.7, 0.9]
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
            im = imread(test_img_ds.Files{1});
            im = insertObjectAnnotation(im,"rectangle", ...
                detectionResults.Boxes{1}, detectionResults.Scores{1});
            figure('Name',"Threshold " + threshold);
            imshow(im);
        end
    end
end

