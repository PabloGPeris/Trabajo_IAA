%https://es.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
    
%% Accuracy del 95.6, en la web pone de pasarle 28*28, he pasado 32*32


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
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%%

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',10, ...
    'Verbose',false, ...
    'Plots','training-progress');

imdsTrain = imageDatastore("C:\Users\luisb\Desktop\MasterAutomaticayRobotica\1. Inteligencia Artificial Aplicada\Trabajo\ImagenesDeep\lenet\Train","IncludeSubfolders",true,"LabelSource","foldernames");

net = trainNetwork(imdsTrain,layers,options);

%% Testeo
label_test = load ('../../ImagenesDeep/lenet/label_test.mat');

for i = 1 : 2000

data_test = imread(sprintf('../../ImagenesDeep/lenet/Test/FIG%d.jpeg',i));

class(1,i) = classify (net, data_test);

end

clase = double(class) - 1;

accuracy = sum(clase == label_test.label_test) /numel(label_test.label_test);

disp ("accuracy: ");
disp(accuracy*100);

% Confussion matrix

C = confusionmat(label_test.label_test,clase);

confusionchart(C);





