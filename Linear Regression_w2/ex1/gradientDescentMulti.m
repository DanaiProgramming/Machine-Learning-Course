function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    temp=zeros(size(X,2),1);
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
%     l=theta'*X';
%     k=l'-y;
%     P=k'*X(:,2);
%     temp1=alpha*(P/m);
% 
%      theta(1,1)=theta(1,1)-temp0;
%      theta(2,1)=theta(2,1)-temp1;

    l=theta'*X';
    k=l'-y;
    P=sum(k);
%     temp0=alpha*(P/m);
%     temp(1,1)=alpha*(P/m);
    temp(1,1)=theta(1,1)-(alpha*(P/m));

    
     for j=2:size(X,2)
         
            l=theta'*X';
            k=l'-y;
            P=k'*X(:,j);
%             temp(1,j)=alpha*(P/m);
            temp(j,1)=theta(j,1)-(alpha*(P/m));
            
     end
     
%      theta(1,1)=theta(1,1)-temp(1,1);
        theta(1,1)=temp(1,1);
     

     for j=2:size(X,2)
%               theta(j,1)=theta(j,1)-temp(j,1);
            theta(j,1)=temp(j,1);
     end
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    
%     keyboard;
end

end
