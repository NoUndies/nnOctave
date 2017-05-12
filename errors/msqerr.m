function [J err] = msqerr(CALCULATED, ACTUAL)
  
  % catch errors and exit if topology doesn't match
  if size(CALCULATED) ~= size(ACTUAL)
    disp('ERROR ~MSQERR: calculated and actual vectors different sizes.')
    break
  end
  
  % calculate absolute error and average squared error
  err = CALCULATED-ACTUAL;
  J = sum(sum(err.^2))/2/size(CALCULATED,2);
end