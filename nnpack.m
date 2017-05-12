function THETAs = nnpack(_THETA, TOPOLOGY)
  L=size(TOPOLOGY,2);
    count=1;
  for i=2:L,
    n = TOPOLOGY(i);
    j = TOPOLOGY(i-1);
    _TMP = reshape(_THETA(count:(count+n*(j+1)-1)),n,j+1);
    THETAs{i-1} = _TMP;
    count = count+n*(j+1);
  end
end