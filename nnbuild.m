function [THETAs Xs]= nnbuild(TOPOLOGY)
  % this function builds a fully connected neural network based on an input
  % topology vector, and returns a structure that contains the nn layers
  % -- Sean Morrison, 2017
  
  % initialize weight matrices based on topology
  N = max(TOPOLOGY);
  L = size(TOPOLOGY,2);
  THETAs = cell(L-1,1);
  Xs = cell(L-1,1);
  
  % step through each layer and generate random theta weightings, create output
  % matrices initialized to zero
  for i=2:L,
    TMP = [];
    for j=1:TOPOLOGY(i),
      TMP = [TMP; rand(1,TOPOLOGY(i-1)+1)];
    end
    THETAs(i-1) = TMP;
    Xs(i-1) = zeros(TOPOLOGY(i),1);
  end
end