
%%% CARGA DE LAS SEÑALES ECG Y LAS ETIQUETAS A PARTIR DE FICHERO .CSV %%%

% Las señales ECG se encuentran en el directorio "Sujetos"
% El fichero .csv contiene los nombres de los ficheros de los sujetos y su clase (userX)
% Se crea una tabla con los nombres de los ficheros de los sujetos y su clase

ref = 'dataset.csv';
tbl = readtable(ref,'ReadVariableNames',false);
tbl.Properties.VariableNames = {'Filename','Class'};

% Se carga cada fichero en la tabla guardando su señal ECG
for i = 1:height(tbl)
    fileData = load(['./Sujetos/',[tbl.Filename{i},'.mat']]);
    % Emplearemos sólo el canal 1 de la señal ECG de los 2 disponibles
    tbl.Signal{i} = fileData.val(1,:); 
end

% Preparación de los datos para acoplarse a la red LSTM
% Signals: cell array 1xn de cada usuario
% Labels: categorical array de cada usuario
Signals = tbl.Signal;
Classes = categorical(tbl.Class);

% Guardar estas variables en un fichero .MAT
save ECGSignalsData.mat Signals Classes


