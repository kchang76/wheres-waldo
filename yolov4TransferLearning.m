function [detector,info] = yolov4TransferLearning(networkName, className, anchorBoxes, trainingData, trainingOption)
% Function for transfer learning
%   networkName = Name of pretrained neural network (Default to ResNet-50?)
%   className = Name of object class to detect
%   anchorBoxes = Cell array of detection heads
%   trainingData = Data for training detector
%   trainingOption = Training parameters (refer to "trainingOptions")
%   detector = Newly trained YOLO v4 object detector
%   info = Detector info

% Set base network
basenet = imagePretrainedNetwork(networkName);

% Display network architecture
% analyzeNetwork(basenet);

% Initialize and normalize image input layer
imageSize = basenet.Layers(1).InputSize;
layerName = basenet.Layers(1).Name;
newInputLayer = imageInputLayer( ...
    imageSize, ...
    Normalization="none", ...
    Name=layerName);

% Replace image input layer with new input layer
dlnet = replaceLayer(basenet, layerName, newInputLayer);

% Create YOLO v4 object detector
featureExtractionLayers = ["activation_22_relu","activation_40_relu"];
detector = yolov4ObjectDetector( ...
    dlnet, ...
    className, ...
    anchorBoxes, ...
    DetectionNetworkSource = featureExtractionLayers);

% Train new network
[detector, info] = trainYOLOv4ObjectDetector( ...
    trainingData, ...
    detector, ...
    trainingOption);

end