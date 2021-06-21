
%% REDES LSTM PARA SISTEMA DE AUTENTICACIÓN BIOMÉTRICA BASADO EN ECG %%
% Roberto Hernández Ruiz
% TFM Máster en Internet de las Cosas 2020-2021

%% 1. Carga de los sujetos: lectura del fichero .mat con las señales y obtención de señales ECG

if ~isfile('ECGSignalsData.mat')
    % Lectura del fichero csv con los sujetos y carga de las señales
    ReadECGData      
end
% Fichero .mat con Señales y Clases asociadas
load ECGSignalsData

% El número de sujetos será el número de clases
numSujetos = size(Classes,1); 

for j=1:numSujetos
    
   % Crear una variable para cada usuario userN = 1xn muestras de señal ECG
   eval(sprintf('user%d = Signals{j};',j));
   
   % Representación de las señales ECG de los sujetos
%    figure(1)
%    subplot(round(numSujetos/2),2,j)
%    plot(Signals{j})
%    title(strcat('User',{' '},num2str(j)))
%    xlim([4000,5200])
%    xlabel('Samples')
%    ylabel('ECG (mV)')  
  
end

%% 2. Segmentación o enventanado de las señales ECG
% Dividir la señal en ventanas de 9000 muestras cada una para estandarizar:
% función segmentSignals de Matlab-> aumento del nº de instancias para entrenar
[Windows,Labels] = segmentSignals(Signals,Classes);

%% Histograma de la longitud de las señales ECG de los sujetos (opcional)

L = cellfun(@length,Signals);
h = histogram(L);
xticks(0:500000:12000000);
xticklabels(0:500000:12000000);
title('Signal Lengths')
xlabel('Length')
ylabel('Number of users')

%% 3. Tras aumentar el número de instancias por usuario para la entrada de la red,
% estandarización del nº de ventanas o instancias para todas las clases

numWindows_Train = [];
numWindows_Test = [];

for j=1:numSujetos
    
    % Ventanas de señal (userX) para cada usuario con sus etiquetas asociadas (userY)
    userX{j,1} = Windows(Labels==(strcat('User',num2str(j))));
    userY{j,1} = Labels(Labels==(strcat('User',num2str(j))));
    
    % Numero de instancias/ventanas generadas para cada usuario
    windowsByUser(j) = size(userX{j,1},1);
    
    % Dividir las instancias de cada sujeto en train (80%) y test (20%)
    [trainIndUser{j,1},~,testIndUser{j,1}] = dividerand(windowsByUser(j),0.8,0,0.2);
    
    % Numero de ventanas que resultan para Train y para Test tras el split
    nTrain = cell2mat(trainIndUser(j));
    nTest = cell2mat(testIndUser(j));
    
    % Señal ECG (userX) y etiqueta (userY) de cada usuario
    x = userX(j);
    y = userY(j);
    
    % Separar 0.8n ventanas (nTrain) para Train y 0.2n ventanas (ntest)
    % para Test, siendo n las ventanas de cada usuario
    XTrainUser{j,1} = x{1}(nTrain);
    YTrainUser{j,1} = y{1}(nTrain);
    XTestUser{j,1} = x{1}(nTest);
    YTestUser{j,1} = y{1}(nTest);
    
    % Concatenar el numero de ventanas de cada usuario en Train y Test
    numWindows_Train = [numWindows_Train size(XTrainUser{j},1)];
    numWindows_Test = [numWindows_Test size(XTestUser{j},1)];
    
end

% Obtener el minimo numero de ventanas de todos los usuarios en Train y Test
minTrain = min(numWindows_Train);
minTest = min(numWindows_Test);

%% 4. Preparación de los datos (XTrain, XTest) y sus etiquetas (YTrain, YTest) para la red LSTM

XTrain = [];
YTrain = [];
XTest = [];
YTest = [];

for j=1:numSujetos
    
    % Obtener los conjuntos completos de Train y Test considerando las
    % mismas ventanas para todos los sujetos (el minimo de ventanas) 
    XTrain = [XTrain XTrainUser{j}(1:minTrain)];
    YTrain = [YTrain YTrainUser{j}(1:minTrain)];
    
    XTest = [XTest XTestUser{j}(1:minTest)];
    YTest = [YTest YTestUser{j}(1:minTest)];
    
end

% Convertir las matrices a vectores columna para ajustarnos a la red LSTM
XTrain = reshape(XTrain,[],1);
YTrain = reshape(YTrain,[],1);
XTest = reshape(XTest,[],1); 
YTest = reshape(YTest,[],1); 

%% 5. Definir la arquitectura de la red LSTM

layers = [ ...
    sequenceInputLayer(1) % señales de entrada unidimensionales
    bilstmLayer(100,'OutputMode','last') % mapear señales a 100 ccticas 
    fullyConnectedLayer(numSujetos) % n usuarios = n clases
    softmaxLayer
    classificationLayer
    ]

% Técnica optimización ADAM
options = trainingOptions('adam', ...
    'MaxEpochs',10, ... % recorrer 10 veces el conjunto de datos completo
    'MiniBatchSize', 150, ... % trabajar con 150 señales a la vez
    'InitialLearnRate', 0.01, ... % coeficiente de aprendizaje
    'SequenceLength', 1000, ... % fragmentar la señal
    'GradientThreshold', 1, ... % evitar gradientes demasiado largos
    'ExecutionEnvironment',"auto",... 
    'plots','training-progress', ... % mostrar gráfica training progress (accuracy vs iteraciones)
    'Verbose',false); 

%% 6. Entrenamiento de la red LSTM

net = trainNetwork(XTrain,YTrain,layers,options);


%% 7. Clasificar señales de Train, visualizar la accuracy y la matriz de confusión

trainPred = classify(net,XTrain,'SequenceLength',1000);
LSTMAccuracy_Train = sum(trainPred == YTrain)/numel(YTrain)*100

figure
confusionchart(YTrain,trainPred,'ColumnSummary','column-normalized','RowSummary','row-normalized','Title','Confusion Chart for LSTM - Train');

%% 8. Clasificar señales de Test

testPred = classify(net,XTest,'SequenceLength',1000);
LSTMAccuracy_Test = sum(testPred == YTest)/numel(YTest)*100

figure
confusionchart(YTest,testPred,'ColumnSummary','column-normalized','RowSummary','row-normalized','Title','Confusion Chart for LSTM - Test');

%% 9. APROXIMACIÓN 2 (EXTRACCIÓN DE CARACTERÍSTICAS Y ANÁLISIS TIEMPO-FRECUENCIA)

% Frecuencia de muestreo (Hz)
fs = 300;

% 2 momentos de freq temporal: freq instantánea y entropía espectral
for j=1:numSujetos
    
    [instFreqUser{j,1},tUser{j,1}] = instfreq(eval(sprintf('user%d',j)),fs);
    [pentropyUser{j,1},t2User{j,1}] = pentropy(eval(sprintf('user%d',j)),fs);
end

%% 10. Calcular instfreq y pEntropy a cada celda en los conjuntos de datos
% Guardar en variables distintas para luego concatenar

instfreqTrain = cellfun(@(x)instfreq(x,fs)',XTrain,'UniformOutput',false);
instfreqTest = cellfun(@(x)instfreq(x,fs)',XTest,'UniformOutput',false);

pentropyTrain = cellfun(@(x)pentropy(x,fs)',XTrain,'UniformOutput',false);
pentropyTest = cellfun(@(x)pentropy(x,fs)',XTest,'UniformOutput',false);

%% 11. Concatenar las características en los mismos conjuntos

% Ahora señal con 2 dimensiones: instFreq (fila 1) y pEntropy (fila 2) por
% cada ventana (tamaño 2 dim x 255 valores de 255 ventanas temporales)
XTrain2 = cellfun(@(x,y)[x;y],instfreqTrain,pentropyTrain,'UniformOutput',false);
XTest2 = cellfun(@(x,y)[x;y],instfreqTest,pentropyTest,'UniformOutput',false);

% Conocer la media de cada característica para cada usuario
% Se observa que las magnitudes difieren en orden de magnitud
for j=1:numSujetos
    meanInstFreqUser{j,1} = mean(instFreqUser{j,1});
    meanPEntropyUser{j,1} = mean(pentropyUser{j,1});
end

%% 12. Estandarización de los datos

XV = [XTrain2{:}];
mu = mean(XV,2);
sg = std(XV,[],2);

% Nuevos conjuntos de entrenamiento XTrainSD y de test XTestSD estandarizados
XTrainSD = XTrain2;
XTrainSD = cellfun(@(x)(x-mu)./sg,XTrainSD,'UniformOutput',false);

XTestSD = XTest2;
XTestSD = cellfun(@(x)(x-mu)./sg,XTestSD,'UniformOutput',false);

%% 13. Medias de las características t/freq extraídas de la señal

instFreqNSD = XTrainSD{1}(1,:); %primera fila es instFreq
pentropyNSD = XTrainSD{1}(2,:); %segunda fila es pentropy

mean(instFreqNSD)
mean(pentropyNSD)


%% 14. Modificar la arquitectura de la nueva red LSTM

layers2 = [ ...
    sequenceInputLayer(2) % ahora señales de entrada bidimensionales (instFreq y pentropy)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(numSujetos)
    softmaxLayer
    classificationLayer
    ]

% Nuevas opciones de entrenamiento: aumentar MaxEpochs
options2 = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

% Por la menor longitud de las señales de entrada (ahora ccticas, antes
% datos puros de la señal) se prevee un entrenamiento mucho más corto

%% 15. Entrenamiento de la nueva red LSTM con las ccticas tiempo-freq

net2 = trainNetwork(XTrainSD,YTrain,layers2,options2);

%% 16. Clasificar señales de Train, visualizar la accuracy y la matriz de confusión

trainPred2 = classify(net2,XTrainSD);
LSTMAccuracy_Train_2 = sum(trainPred2 == YTrain)/numel(YTrain)*100

figure
confusionchart(YTrain,trainPred2,'ColumnSummary','column-normalized','RowSummary','row-normalized','Title','Confusion Chart for LSTM - Train 2');

%% 17. Clasificar señales de Test, visualizar la accuracy y la matriz de confusión

testPred2 = classify(net2,XTestSD);
LSTMAccuracy_Test_2 = sum(testPred2 == YTest)/numel(YTest)*100
figure
confusionchart(YTest,testPred2,'ColumnSummary','column-normalized','RowSummary','row-normalized','Title','Confusion Chart for LSTM - Test 2');







