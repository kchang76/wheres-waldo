function utils = data_utils()
    utils = {};
    utils.augment = @augment;
    utils.flatten_ds = @flatten_ds;
    utils.calculate_anchors = @calculate_anchors;
    utils.train_test_split = @train_test_split;
    utils.train_validation_test_split = @train_validation_test_split;
    utils.get_metrics_report = @get_metrics_report;
    utils.chunk_image = @chunk_image;
    utils.chunk_image_ds = @chunk_image_ds;
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
    for i=1:length(ds_collection)
        ds_collection{i} = ds_collection{i}.UnderlyingDatastores{1};
    end

    image_ds = ds_collection{1}.UnderlyingDatastores{1};
    for i=2:length(ds_collection)
        sub_image_ds = ds_collection{i}.UnderlyingDatastores{1};
        for j=1:length(sub_image_ds.Files)
            image_ds.Files{end+1} = sub_image_ds.Files{j};
        end
    end

    Waldo=[];
    bl_ds_tbl = table(Waldo);
    for i=1:length(ds_collection)
        sub_bl_ds = ds_collection{i}.UnderlyingDatastores{2};
        for j=1:length(sub_bl_ds.LabelData)
            Waldo = {sub_bl_ds.LabelData{j,1}};
            subtable = table(Waldo);
            bl_ds_tbl = [bl_ds_tbl;subtable];
        end
    end

    bl_ds = boxLabelDatastore(bl_ds_tbl);
    flat_ds = combine(image_ds,bl_ds);
    flat_ds = shuffle(flat_ds);
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

function chunked_ds = chunk_image_ds(ds, chunk_size)
    if ~exist('chunk_size','var')
          chunk_size = 128;
    end
    img_ds = ds.UnderlyingDatastores{1};
    bl_ds = ds.UnderlyingDatastores{2};

    mkdir(sprintf("dataset/%i/full_scene_ds", chunk_size));
    delete(sprintf("dataset/%i/full_scene_ds/*.jpg", chunk_size));

    chunk_bl_table = table({[0,0,0,0]},'VariableNames',string(bl_ds.LabelData{1,2}));
    chunk_bl_table(1,:) = [];
    current_folder = pwd;

    for file_index=1:length(img_ds.Files)
        filepath = img_ds.Files{file_index};
        [~,filename,~] = fileparts(filepath);
        im = imread(filepath);
        im_size = size(im);

        % Save image as new subsections of chunk_size by chunk_size
        for row_index=1:ceil(im_size(1)/chunk_size)
            r_end = min((row_index)*chunk_size, im_size(1));
            r_start = r_end - chunk_size + 1;
    
            for col_index=1:ceil(im_size(2)/chunk_size)
                c_end = min((col_index)*chunk_size, im_size(2));
                c_start = c_end - chunk_size + 1;
   
                % check if boundary label exists for file and if it is in
                % this section (at least 40% of the area of boundary is in
                % section
                if (any(bl_ds.LabelData{file_index,1} ~= [0 0 0 0]))
                    [overlap, intersect] = area_overlap([c_start, r_start, chunk_size], bl_ds.LabelData{file_index,1});
                    if overlap >=0.4                        
                        % write image subsection to file (so we can debug later)
                        im_subsection = im(r_start:r_end, c_start:c_end, :);
                        imwrite(im_subsection, sprintf("dataset/%i/full_scene_ds/%s_%i_%i.jpg", chunk_size,filename,row_index,col_index));
                    
                        % store bounding label
                        chunk_bl_table = [chunk_bl_table;{intersect;}];
                    end
                end
            end
        end
    end

    chunk_img_ds = imageDatastore(sprintf("dataset/%i/full_scene_ds", chunk_size));
    chunk_bl_ds = boxLabelDatastore(chunk_bl_table);
    chunked_ds = combine(chunk_img_ds,chunk_bl_ds);
end


function [overlap_percent, intersection_points] = area_overlap(square, rect)
    overlap_percent = 0.0;
    intersection_points = [0 0 0 0];

    x1 = square(1);
    y1 = square(2);
    s_length = square(3);
    s = polyshape([x1, x1, x1+s_length-1, x1+s_length-1], [y1, y1+s_length-1, y1+s_length-1, y1]);

    x2 = rect(1);
    y2 = rect(2);
    r_xlength = rect(3);
    r_ylength = rect(4);
    r = polyshape([x2, x2, x2+r_xlength-1, x2+r_xlength-1], [y2, y2+r_ylength-1, y2+r_ylength-1, y2]);

    intersection = intersect(s, r);
    if intersection.NumRegions == 0
        return;
    else
        overlap_percent = area(intersection) / area(r);

        x = min(intersection.Vertices(:,1));
        y = min(intersection.Vertices(:,2));
        x_length = max(intersection.Vertices(:,1)) - x + 1;
        y_length = max(intersection.Vertices(:,2)) - y + 1;

        % put x and y coordinates in sub-image context
        x = x - x1 + 1;
        y = y - y1 + 1;

        intersection_points = [x y x_length y_length];
    end
end
