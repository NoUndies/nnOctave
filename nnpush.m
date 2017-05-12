function _Z = nnpush(_THETA, _X)
  _X = [ones(1,size(_X,2)); _X];
  _Z = _THETA*_X;
end