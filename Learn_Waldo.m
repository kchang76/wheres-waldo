% Quick script to load in Waldo training data, then train and test a model

%% Specify Training Data Inputs

% gTruth is the output of MATLAB's imageLabeler, export as table
gTruth = load("gTruth.mat");

% First column of table is image filenames, second column are bounding
% boxes
imds = imageDatastore(gTruth{:,1});
blds = boxLabelDatastore(gTruth(:,2));

% Cascaded object detector will require positive and negative training sets
positiveInstances = combine(imds,blds);
%negativeFolder = insert filepath to negative images here
%% Train Cascaded Object Detector (Viola-Jones)
fprintf("Training...start!");
trainCascadeObjectDetector('waldodetector.xml',positiveInstances,negativeFolder,FalseAlarmRate=0.01,NumCascadeStages=3);
fprintf("Training done");
%% Test the detector
% Load the trained detector
detector = vision.CascadeObjectDetector('waldodetector.xml');

% Read in the scene to find Waldo in
testIm = imread("data/windowed/scene1/1-3.png");

% Find and annotate all "Waldos", show the result image w/ bboxes
bbox = step(detector,testIm);
detectedImg = insertObjectAnnotation(testIm,'rectangle',bbox,'Waldo');
figure;
imshow(detectedImg);