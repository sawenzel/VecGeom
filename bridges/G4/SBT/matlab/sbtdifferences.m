function res = sbtdifferences(values, first, count)
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
           if (current < 1000)
                disp (['Different point found, index ' int2str(i+first-1) ' difference is ' num2str(difference)]);
           end
       end;
   end    
   res = res(1:current,:);
end
