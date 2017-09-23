function costs_interpolated = sceneStats(data, num_objects, time_axis)
  expands = data(:,1);
  times = cumsum(data(:,2));
  costs = data(:,3) / (2.0 * num_objects);
  costs_interpolated = interp1(times, costs, time_axis, 'previous');
  % plot(xq, costs_interpolated)
  
  
  


