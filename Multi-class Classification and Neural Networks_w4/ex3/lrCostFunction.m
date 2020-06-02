function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% temp1=sum(theta.*2);
% temp2=lambda/(2*m);
% regularizationVar=temp2*temp1;
% 
% h=sigmoid(X*theta);
% var1=-y*log(h)';
% var2=(1-y)*log(1-h)';
% var3=sum(var1-var2);
% J=(var3./m)+regularizationVar;


h=sigmoid(X*theta);
% keyboard;
k=log(h')*(-y);
l=log(1-h')*(1-y);
p=k-l;
J1=(1/m)*p;

o=theta(2:size(X, 2),1).^2;
o=sum(o);
J2=(lambda/(2*m))*o;

J=J1+J2;



% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 


% var1=sigmoid(X)-y;
% var2=X'*var1;
% grad=var2/m;
% 
% tempArray=theta;
% tempArray(1)=0;
% grad=grad+ (lambda*tempArray)/m;


grad=((h-y)'*X)/m;

grad(2:end)=grad(2:end)'+((lambda*theta(2:end))./m);


% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



% =============================================================

grad = grad(:);

end
