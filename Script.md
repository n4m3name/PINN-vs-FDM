# Intro: Convergence and Accuracy of PINNs vs. Classical Finite Difference Schemes for BVPs

- The general/personal motivation: Apply neural nets to everything
  - Interesting object
  - Opportunity to investigate: not feasibility, but appropriateness
    - Accuracy, efficiency
    - Compared to known, proven methods (numerical, FDM)
  - Formulating the research question: How do NN's compare to standard difference methods for solving BVP's

## Prior work
- Raissi from Brown Universiry coined the term PINN's
  - Showed they work
- Covered 2 things in paper:
  - Forward: NN's can approximate PDE solutions by minimizing the residual at random points, without a mesh
  - Inverse: Showed PINNs can recover unknown PDE parameters from noisy observations of the solution
- Wang from University of Pennsylvania showed their limitations
  - Have a certain accuracy threshold
  - Showed more than this (proposed a solution), but keeping it brief
    - Gradient descent can converge to local, not global minima
      - The loss function is generally not perfectly convex
    - Changing one weight affects output everywhere
    - "Conflict"
      - Ex: A gradient step that reduces the residual at x=.3 might increase it at x=.6
        - Updating for either would pull the other in the opposite direction
    - Compare to FDM's, where fixing a solution at one point does not affect another; no "global tug of war"
- Peridikaris co-authored both papers
- Rahaman's spectral bias (known issue w NN's w GD): even if the optimizer is working perfectly (ie no stalling on local minima), the network preferentially fits smooth components
  - Leads to the "fuzzy" solutions NN's are known for
  - Guitar analogy: Easy to get the general notes, perfect tune takes longer
- Begs the question: Why use gradient descent?
  - Asked claude: Claude told me "Least bad option".
  - Ex: If we want finer detail, we may as well solve it using a different method altogether.
  - Tradeoff: Generality/ease of use vs fine details
    - Mesh-free (ie can literally pick a few points and learn a solution)
      - Irregular domains: How do meshes handle complex geometry, moving boundaries, etc?
    - Dimensionality: Scales w parameter count, not grid size
    - Inverse problems: To my knowledge, FDM have no mechanism

## Test problems
- Start analysis
- Taken from the Ch.4 - BVP slides

- 1: Poisson ODE, linear, constant coefficient
- 2: Variable coefficient linear ODE
- 3: Nonlinear, no closed form solution
  - Note: Chose the tolerance to be orders of magnitude smaller than both the PINN and the FDM solution

- Why these
  - Already have exposure, derived solutions, and established convergence theory from class (allows us to take these for granted, save a bit of time during the presentation)
  - Compare performance on linear vs. nonlinear, variable vs. constant coefficients, exact vs reference solution.

SELF NOTE: POISSON = RHS ONLY DEPENDS ON X

## FDM Implementation (Review)
- Follows course notes, outputs results
- Mesh: Splitting the interval of interest on $n$ interior grid points
- Compute centered dif for problems 1 and 3, use symmetric differencing for problem 2. 
- Gives a tridiagonal system that is solved in O(n) 
  - Nonlinear for problem 3, so we iterate as discussed in the next slide
- As we saw in class, when applying the BVP convergence theorem, we obtain a convergence rate of r=2 and an error that is quartered each time the number of grid points doubles
  - Key: The error reliably decreases with larger $n$
- Condition number calculated to give us a predictable accuracy ceiling
  - Floating point amplification when solving the linear system

SELF NOTE: WHY TRIDIAGONAL = O(n) NOT O(n^3)?  
- Full matrix: 
  - eliminating row i updates every row below it (n-i rows), each with n entries.  
- Tridiagonal: 
  - row i only has 3 nonzero entries, so elimination only affects row i+1 (the only row below that overlaps), touching 3 entries.
  - One operation per row, n rows, done.


## Nonlinear solvers
- Since problem 3 has no closed-form solution, we use iterative solvers
- Again, we use exactly those covered in lecture:
- First shooting methods, which reduce boundary problem to initial value problem
  - For shooting methods, we used Newton shooting which converged in 5 iterations and Bisection, which converged in 41.
  - Likewise, for finite difference methods we used Picard FD which converged in 13 iterations and Newton FD, which converged in 2
    - As we saw in class, Newtons convergence is quadratic when the Jacobian is nonsingular (invertible) at the solution and the initial guess (the zero vector) is close to the actual solution (peaks at .05) so in both cases it works extremely well for this problem.
    - 


## PINN Formulation
- Hard boundary conditions
  - If asked: g0, g1 are the bc's u0, u1
  - This part of the equation is the simplest equation that hits both (straight line from one to the other)
  - The NN component, then, is the "correction" which learns/conforms to the solution
  - Note: x(1-x) causes the correction to vanish at either boundary point
- We use mean squared PDE residual (ODE in our case) at each point
  - Note (dnt say): Collocation is picking each point xi, enforing ri=0 at those points only
- To compute derivatives of u, we use pytorch's autograd function which computes each to machine precision
- For gradient descent: Adam
  - Regular: Steps downhill
  - Adam: Tracks "momentum" (running average of past gradients), adapts step size to each parameter individually
    - This helps the network handle flat/noisy regions of the loss landscape
      - Ie momentum should carry it through flat and saddle points
- After Adam runs, L-BGFS
  - Newton's method simulator
    - Uses curvature (second derivative) to measure "distance" to minima (=0?)
    - Full newton is expensive on 3200 parameters, L-BFGS approximates it using the last few gradients instead of the entire curvature surface
    - Guitar analogy: Adam handles the rough tuning, L-BFGS handles the fine tuning
    - Dont ask what it stands for: Limited-memory Broyden-Fletcher-Goldfarb-Shanno
- Tanh activation: differetniable 
- This was the approach used in Raissi's 2019 paper and worked well

- Add as necessary: Each neuron computes the weighted sum of the inputs, applies activation function
$$
z_j^{(l)} = \sigma!\left(\sum_k w_{jk}^{(l)} , z_k^{(l-1)} + b_j^{(l)}\right) 
$$
  - $z_k^{(l-1)}$ = outputs from the previous layer (inputs to this neuron)                                          
  - $w_{jk}^{(l)}$ = weights (how much each input matters)                                                           
  - $b_j^{(l)}$ = bias (offset)                                                                                      
  - $\sigma$ = activation function (tanh in our case)


Self note (do NOT say): ri is the pde but w everything moved to one side, =0



### PINN's as Nonlinear Collocation
- Do note say: Collocation is picking each point xi, enforing ri=0 at those points only
- Classical: Choose a basis (e.g., polynomials/b-splines), write the solution as a linear combination, enforce the PDE residual = 0 at chosen points.
- PINN: We do what we just covered
- Key difference: Number of collocation points ($N_c$) and parameter count (Deg of freedom, 3200 for our network) are independent: We have a fixed number of parameters for out network
- In FDM, we know that increasing the resolution of the mesh leads to less error due to increasing accuracy of the finite difference approximation (ie "error / 4 when $n$ doubles)
- But: increasing resolution for the NN simply checks more points and does not necessarily result in a more accurate solution: Our derivatives are already computed to machine precision and we don't get more parameters; 
- Already discussed known limitations
  - Spectral bias
  - Non-convex loss
- Again, this is the optimizer bottleneck: gradient descent has a limit
- In essence: 
  - For FDM's the error reliably shrinks ie $r \rightarrow 2$
  - For PINN's: The error is approximately zero

Note: $r = \frac{\log(e_{n} / e_{2n})}{\log(2)}$
- $r=2$ means double $n$, error drops by $2^2$
- the error is $O(h^2)$ and $h = 1/(n+1)$. So:
  - At grid size $n$: error $\approx C \cdot h^2$
  - Double $n$: $h$ halves, so error $\approx C \cdot (h/2)^2 = C \cdot h^2/4$ 

### Results: FDM convergence
- Exactly as expected: r=2
- Error caclualtion process (if asked)
1. Solve the FDM system to get the approximate solution $U$
2. Evaluate the exact solution at the same points (or the reference solution for Problem 3)
3. Error = $\max |U_i - u_{\text{exact}}(x_i)|$ (the $L^\infty$ norm)
4. Compute the rate from consecutive errors  
   - ie for $n=31$: error = 7.79e-4, $n=63$: error = 1.92e-4
   - $r = \frac{\log(7.79\text{e-}4 ;/; 1.92\text{e-}4)}{\log(2)} = \frac{\log(4.06)}{\log(2)} \approx 2.02$

### Results: Accuracy vs smapling density
- FDM: Decreases linearly w increasing Nc
- PINN: Plateaus at around 10^-5

### Results: PINN Plateau
- Errors stop at around 10^-5
  - This is variable and not really theoretically-driven
  - Ex using pure gradient descent would likely result in a higher error
- Convergence rate \approx 0 (flat curve)
- Exactly as we expect

### Results: Cost
- Including training, PINN is incredibly slow compared to FDM
- Not including training (ie a pre-trained model) would be roughly similar

### Results: Condition vs loss ladnscape
- Review: Condition number measures floating point roundoff amplification when solving the linear system $A_h U = F$
  - ie Machine epsilon is about 10^-16, when you solve the system this number is multiplied by $\kappa \approx \frac{4n^2}{\pi^2}$
    - Ex: At $n$ (num grid points) = 1023
      - $\kappa \approx 4 \times 10^5$
      - Bet possible accuracy: $\kappa \cdot \epsilon_{\text{mach}} \approx 4 \times 10^5 \times 10^{-16} = 4 \times 10^{-11}$
- For FDM we have a predictable, computable error
- For NN's we have a sort of loose, variable error;
  - Not that we cannot compute it, but it is not a theoretical guarantee


### Conclusions:
- FDM performs better that neural networks for well-posed 1D BVP's
- The NN accuracy plateau is consistent with existing findings
- There are applications where a PINN would outperform FDM's, we leave this as a topic for further research.