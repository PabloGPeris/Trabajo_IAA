


deepNetworkDesigner;
net = googlenet;
analyzeNetwork(net);





layers = [ imageInputLayer([24 24 1])
convolution2dLayer(3,8,'Padding','same’)
batchNormalizationLayer reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,16,'Padding','same’)
batchNormalizationLayer reluLayer
maxPooling2dLayer(2,'Stride’,2)
fullyConnectedLayer(10) softmaxLayer classificationLayer];




options = trainingOptions('sgdm', 'InitialLearnRate',0.01, 'MaxEpochs',4,
'Shuffle','every-epoch', 'ValidationData',imdsValidation,
'ValidationFrequency',30,'Verbose',false, 'Plots','training-progress');


images 


net = trainNetwork (imagesTrain, layers, options);

class = classify (net, imagesTest);

accuracy = sum(class == labelTest)/numel(labelTest)
