function Xs = nnfeedforward(THETAs, Xs, INPUT)
  % this function feeds all inputs forward through the network. Full
  % connectivity is assumed. This function returns weightings theta, and
  % output values A at all nodes through all layers, for all input samples.
  % -- Sean Morrison, 2017
  
  % feed input into first layer
  _THETA = cell2mat(THETAs(1));
  _Z = nnpush(_THETA,INPUT);
  Xs(1) = sigmoid(_Z,0);
  
  % loop through remaining cells to feed forward through network
  length = size(THETAs,1);
  running = true;
  while running
    i = 2;
  
    _THETA = cell2mat(THETAs(i));
    _X = cell2mat(Xs(i-1));
    _Z = nnpush(_THETA,_X);
    Xs(i) = sigmoid(_Z,0);
    i = i+1;
    
    if i = length,
      running = false;
    end
  end
end