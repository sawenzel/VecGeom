
% evaluates from vector and plots differences 
%
% this function is used internally by sbtplot and sbtplot3d, users do not
% have to run it by their own ...

function res = sbtdifferences(values, first, count, points, directions)
   current = 0;
   if count < 0
      loopcount = length(values);
   else
      loopcount = count;
   end
   res = values;
   for i = 1:loopcount
       difference = values(i);
       if (length(difference) == 3)
            difference = norm(difference);
       end
       if abs(difference) > 1e-6
           current = current+1;          
           res(current) = i;
           if (current < 100)
                index = i+first-1;
                disp ([int2str(current) '. Different point found, index ' int2str(index) ' difference is ' num2str(difference)]);
                if (nargin > 4)
                    point = points(index,:);
                    direction = directions(index,:);
                    disp (['    point = Vector(' printvec(point) ');']);
                    disp (['    direction = Vector(' printvec(direction) ');']);
                end
           end
       end;
   end    
   res = res(1:current,:);
end

function res=printvec(p)
    prec = 16;
    res = [num2str(p(1),prec),', ' num2str(p(2),prec),', ' num2str(p(3),prec)];
end    
