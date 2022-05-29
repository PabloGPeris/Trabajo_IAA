

%deepNetworkDesigner

%% Capas
layers = [
    imageInputLayer([32 32 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLaye
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% Datos

imdsTrain = imageDatastore("C:\Users\luisb\Desktop\MasterAutomaticayRobotica\1. Inteligencia Artificial Aplicada\Trabajo\ImagenesDeep\lenet\Train","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.8);

augimdsTrain = augmentedImageDatastore([32 32 1],imdsTrain);
augimdsValidation = augmentedImageDatastore([32 32 1],imdsValidation);


%% Opciones entrenamiento

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',10, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    "ValidationData",augimdsValidation);


net = trainNetwork(imdsTrain,layers,options);

