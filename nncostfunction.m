function [J grad] = nncostfunction(_THETA,THETAs,Xs,INPUT,OUTPUT,TOPOLOGY,errfunc,ACTFNS,lambda)
  % this function takes in a vector of theta values, and calculates the cost
  % function of the network based on a given error function.
  % -- Sean Morrison, 2017
  
  % get size parameters and create 
  m=size(INPUT,2);
  L=size(TOPOLOGY,2);
  
  % use topology to repack thetas into our NN, so that we can use nnfeedforward
  THETAs = nnpack(_THETA,TOPOLOGY);
  
  % feedforward to calculate all output values through the network
  Xs = nnfeedforward(THETAs,Xs,INPUT,ACTFNS);
  
  % evaulate cost function
  [J err] = errfunc(cell2mat(Xs(end)),OUTPUT);
  
  % calculate the regularization value and gradients. We're so fancy, we're going
  % to build the gradient vector backwards (bottom to top).
  count = size(_THETA,1);
  grad = zeros(count, 1);
  reg = 0;
  for i=L:-1:2,
    n = TOPOLOGY(i);
    j = TOPOLOGY(i-1);
    _X = cell2mat(Xs(i-1));
    _Tn = cell2mat(THETAs(i-1));
    if i==L,
      _Xn_1 = cell2mat(Xs(i-2));
      _DLT = err.*ACTFNS{i-1}(nnpush(_Tn,_Xn_1),1);
      _GRAD = _DLT*cell2mat(Xs(i-2))';
      _GRAD = [sum(_DLT,2); _GRAD(:)];
    elseif i==2
      _THETA = cell2mat(THETAs(i));
      _THETA(:,2:end);
      _DLT = _THETA(:,2:end)'*_TMP.*ACTFNS{i-1}(nnpush(_Tn,INPUT),1);
      _GRAD= _DLT*INPUT';
      _GRAD = [sum(_DLT,2); _GRAD(:)];
    else
      _Xn_1 = cell2mat(Xs(i-2));
      _THETA = cell2mat(THETAs(i));
      _DLT = (_THETA(:,2:end)'*_TMP).*ACTFNS{i-1}(nnpush(_Tn,_Xn_1),1);
      _GRAD= _DLT*cell2mat(Xs(i-2))';
      _GRAD = [sum(_DLT,2); _GRAD(:)];
    end
    grad((count-n*(j+1)+1):count) = _GRAD(:);
    count = count - n*(j+1);
    _TMP = _DLT;
    _THETA = cell2mat(THETAs(i-1));
    reg = reg+sum(sum(_THETA(:,2:end).^2));
  end
  
  % final calculation of J and cost function gradient.
  reg = lambda/2/m*reg;
  J = J+reg;
  grad = grad/m;
end