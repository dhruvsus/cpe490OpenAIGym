# 490Project
observation: [position of cart, velocity of cart, angle of pole, rotation rate of pole]


Stopping conditions:
  * pole is > 15 degrees from vertical
  * cart > 2.4 units from center


Solving conditions:
  * average reward of 195.00 over 100 trials


Approach:
  * Simple linear model (weighted sum)
  * Hill-climbing algorithm
  * Policy gradient algorithm
  * Q learning
  * Deep Q learning

d = p1*(position_of_cart) + p2*(velocity_of_cart) + p3*(angle_of_pole) + p4*(rotation_rate)
if d > 0 ==> move right
else ==> move left
