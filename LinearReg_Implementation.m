#Implementation of Gradient Descent algorithm to minimize linear regression cost function
#Evan Li

#pkg install -forge symbolic; 
#pkg load symbolic;

#prelim variables
#warning("off","all"); 

a = 0.007; #Step Size
lim = 0.00001 #Convergence precision limit

#x/y of training set----------------------

trainingx = [1.35;3.56;7.63;5.12;1.00]; #x values of training set
trainingy = [5.32;4.37;2.74;7.32;1.00]; #y values of the training set
constant  = [1;1;1;1;1]; #constant term in linear equation 

fn = [constant,trainingx]; #ax + b

training  = [trainingx,trainingy];

prevt1 = -999999;
prevt2 = -999999;
iterations = 0; 

syms t1;
syms t2;

theta1 = 0; #constant term of linear equation
theta2 = 0; #slope



t = [t1;t2];

#Cost Function f:
g = fn*t-trainingy;
f = (g.*g); #squaring each element in g (we omit the constant term in the front) 

jsum = sum(jacobian(f)); #Sum of the jacobian of matrix f 
partialsubbed = function_handle(jsum);

while(iterations <1000)
  
  temp = partialsubbed(theta1,theta2);
  
  prevt1 = theta1;
  prevt2 = theta2;
  
  theta1 = theta1 - (a)*temp(1);
  theta2 = theta2 - (a)*temp(2);
  
  iterations = iterations+1;

endwhile

  


disp("Results");
disp(double(theta1));

disp(double(theta2))



#Expression:(trainingset*t-y_t).*(trainingset*t-y_t)

#Partial Derivative: diff(sum((trainingset*params-y_t).*(trainingset*params-y_t)),x1 OR x2)






