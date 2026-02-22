
# SDRP

## Background

### Hybrid/Deterministic/stocastic Dynamics


### Gradint Step

### Deterministic / Stochastic dynamics 
In planning problems, the environment dynamics are defined by a known transition function
T(s,a)=s',
which specifies(almost) the next state resulting from applying action a in state s. Unlike reinforcement learning, where the dynamics are typically unknown and learned from interaction, planning methods such as JaxPlan assume access to this transition model (or a differentiable approximation of it).

* Deterministic dynamics arise when the next state is uniquely determined by the current state and action. 

  For example, in a reservoir system with water level $$l$$  , releasing w units of water deterministically yields $$l'=l-w $$

* Stochastic dynamics occur when transitions involve uncertainty, and the next state is drawn from a conditional distribution $$P(s' \mid s,a)$$. For instance, if rainfall $$r \sim \mathcal{N}(\mu,\sigma^2) $$ affects the reservoir, the transition becomes $$ l'=l-w+|r| $$

 * Hybrid dynamics combine deterministic and stochastic components within the state. 
 
   For example, the control action (released water) may be deterministic, while environmental effects such as rainfall are stochastic, resulting in mixed deterministic–stochastic state evolution.
### Discrete / Continuous / Hybrid Dynamics
The transition (dynamics) function can operate over **continuous**, **discrete**, or **hybrid** state spaces.
* **Continuous dynamics** arise when states and actions take values in continuous domains and the transition is described by smooth functions.

  **Example:**  
  A continuous water-level or physical system:
  $$
  x_{t+1} = x_t + u_t - 0.1\,x_t,
  $$
  where both the state  $$ x_t , u_t $$ 
  are real-valued and the transition is differentiable.

* **Discrete dynamics** occur when the transition involves logical conditions or comparisons such as `if`, `and`, `or`, `<`, `=`, or `xor`, resulting in non-continuous state changes.

  **Example:**  
  A binary valve system:
  $$
  v_{t+1} =
  \begin{cases}
  1 & \text{if } x_t > \theta \text{ and } a_t = \text{open}, \\
  0 & \text{otherwise}.
  \end{cases}
  $$
  Here, the next state changes discontinuously based on logical conditions.

* **Hybrid dynamics** combine continuous state evolution with discrete decisions or modes.

  **Example:**  
  A reservoir with a continuous water level 
        $$ l_{t+1} = l_t + r_t - v_t * w,$$
  where $$v_t $$ is determined by a discrete rule (open/closed valve), while the water level evolves continuously.


### Gradint Step


#### Rmsprop

###   Exploration vs Exploition

### On policy & Off policy 

#### Non-stationary
####  Markovian
#### Non-Markovian (History-dependent)
#### Stochastic
#### Deterministic Policy(example)
#### Stochastic Policy (example)
### Open loop & Close loop

###
### 
## Main idea
