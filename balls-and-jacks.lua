
-- Balls and jacks shape transport problem

exec_mode = 0

-- helper function
function insidePolygon(xpoints,ypoints, x,y)
  local oddNodes = false
  local j = #xpoints
  for i = 1, #xpoints do
    if (ypoints[i] < y and ypoints[j] >= y or ypoints[j] < y and ypoints[i] >= y) then
      if (xpoints[i] + ( y - ypoints[i] ) / (ypoints[j] - ypoints[i]) * (xpoints[j] - xpoints[i]) < x) then
        oddNodes = not oddNodes;
      end
    end
    j = i;
  end
return oddNodes end

-- helper function
function ball(x0,y0,r,x,y)
  xr = x-x0
  yr = y-y0
  rsq = xr^2 +yr^2

  if (rsq < r*r) then return true end
  return false
end

-- required functions below
function initial_function(x,y)
  if ball(0.4, 0.4, 0.07, x,y) then return 2.0 end
  if ball(0.4, 0.4, 0.10, x,y) then return 1.0 end

  if ball(0.4, 0.2, 0.03, x,y) then return 3.0 end
  if ball(0.4, 0.2, 0.07, x,y) then return 2.0 end
  if ball(0.4, 0.2, 0.10, x,y) then return 1.0 end

  -- straight cross
  xcross1 = {.270, .270, .120, .120, .090, .090, .020, .020, .090, .090, .120, .120}
  ycross1 = {.300, .330, .330, .460, .460, .330, .330, .300, .300, .230, .230, .300}
  if insidePolygon(xcross1, ycross1, x, y) then return 1.0 end

  -- diagonal cross
  xcross2 = {.0200000, .0624264, .0800000, .0975736, .1400000, .1012131, .2224264, .1800000, .0800000, .0200000, .0200000, .0587868}
  ycross2 = {.030000, .030000, .0475736, .0300000, .0300000, .0687868, .190000, .190000, .0900000, .1500000, .1075736, .0687868}

  if insidePolygon(xcross2, ycross2, x, y) then return 1.0 end
  return 0
end

function velocity_function(x,y)
  return 0.0707106781,0.0707106781
end

function boundary_condition(x,y,t)
  return 0.0
end
