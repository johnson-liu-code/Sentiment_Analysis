


% Don't use this file for anything, it's just a testing ground for math stuff.

function [y] = g(x)
    if x < 100
        y = (x/100)^(3/4);
    else
        y = 1;
    end
end

X = [7/100 1/20 2/70; 3/100 1/30 3/50; 3/100 1/40 4/90];
W = [1/3 2/9 7/9; 1/5 2/5 2/7; 4/7 3/7 7/10];

iter = 200;
eta = 1;
new_W = [0 0 0; 0 0 0; 0 0 0];


for k = 1:iter
    for i = 1:3
        a = [0 0 0];
    
        for j = 1:3
            a = a + g( X(i,j) ) * ( W(i,:) * W(j,:).' - log( X(i,j) ) ) * W(j,:);
        end

        new_W(i,:) = W(1,:) - 2 * a;
    end

    W = new_W;
    %pause()
end