function out = sigmoid(Zs, deriv)
% recursive sigmoid activation function to return both the sigmoid and derivative
  
  if deriv == 1,
    out = sigmoid(Zs,0).*(1-sigmoid(Zs,0));
  else
    out = 1./(1+exp(-Zs));
  end
end