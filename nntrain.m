function THETAs = nntrain(options,THETAs,Xs,INPUT,OUTPUT,TOPOLOGY,ACTFNS,errfn,lambda)
 % this function uses a solver to train the neural network;
 % fmincg has been chosen due to its more efficient use of memory than fminunc.
 % Since fmincg is not a default solver in Octave, it must be included in the
 % script's root folder.
 % -- Sean Morrison, 2017
  
  % define function handle for the cost function
  costfunction = @(P)nncostfunction(P,THETAs,Xs,INPUT,OUTPUT,TOPOLOGY,errfn,ACTFNS,lambda);
  
  % unroll theta matrices into a single vector that can be minimized
  _THETA = [];
  for i = 1:size(THETAs,1),
    _TMP = cell2mat(THETAs(i));
    _THETA = [_THETA; _TMP(:)];
  end
  
  % minimize cost function using fminunc
  [_THETA cost] = fmincg(costfunction, _THETA, options);
  C = [];
  C = [C cost];
  
  
  % pack converged theta values
  THETAs = nnpack(_THETA, TOPOLOGY)
  
  disp('Network trained.')
  pause;
end