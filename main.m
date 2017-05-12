% This is the main script for a fully vectorized neural network implementation,
% written by Sean Morrison, 2017. Instructions for how to initialize and train
% a neural network have been given below. 

% get ready to rock and roll
clear all

addpath(genpath('/home/seanny/Octave Scripts/blaze'))

% create vector for neural network topology: [input L1 L2 ... Ln output]
TOPOLOGY = [2 2 1];

% create cell array for layer activation functions: [L1 L2 ... Ln output]
% can also use, for example, ACTFNS{a:n} = @fcn to build deep networks
ACTFNS = cell(1,size(TOPOLOGY,2)-1);
ACTFNS(1:end) = @sigmoid;

% build network based on topology -- assumes a fully connected network
[THETAs Xs] = nnbuild(TOPOLOGY);

% load training data and test data -- simple counting exercise
INPUT = [1 1 0 0;
         1 0 1 0];
             
OUTPUT = [1 0 0 1];

% specify options for fmincg solver -- in this case, max 1000000 iterations
options = optimset('MaxIter', 100000);

% regularization parameter. In this case, lambda = 0 results in 0 regularization
lambda = 0;

% train network using mean squared error algorithm
THETAs = nntrain(options,THETAs,Xs,INPUT,OUTPUT,TOPOLOGY,ACTFNS,@msqerr,lambda)

% feed test inputs through trained network, determine extent of the error
Xs = nnfeedforward(THETAs,Xs,[1;1],ACTFNS);
Xs(end)
Xs = nnfeedforward(THETAs,Xs,[1;0],ACTFNS);
Xs(end)
Xs = nnfeedforward(THETAs,Xs,[0;0],ACTFNS);
Xs(end)