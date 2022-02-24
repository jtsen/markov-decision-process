# Markov Process (MDP) Solver

## Description
This program consists of a Generic Markov Process Solver (MDP) implemented with Python's built in libraries.


## Usage
To compile and run this program:
```bash
python path [-df $df] [-min] [-tol $tol] [-iter $iter] input-file
```
Example:
```bash
python ./mdp.py -min restaurant.txt
python ./mdp.py -df 0.9 teacher.txt
python ./mdp.py maze.txt
```

Where:
* All of the flags are OPTIONAL.
    * ```[-min]``` toggles minimization of values (values as costs instead of rewards)
    * ```[-df]``` should be followed with a floating point number for setting a custom discount factor
    * ```[-tol]``` should be following by a floating point number for setting a custom tolerance parameter for value iteration
    * ```[-iter]``` should be followed by an integer for setting a custom iteration limit for value iteration
    * [--help] for help.

## Implementation Details
This program will only work with input files in the format that is apparent in the sample text files.

The implementation follows the pseudo-code as provided by Professor Paul Bethe from NYU in the course of CSCI-GA.2560 Artificial Intelligence:

    𝜋 = initial policy (arbitrary)
    V = initial values (perhaps using rewards)
    for {
      V = ValueIteration(𝜋) // computes V using stationery P
      𝜋' = GreedyPolicyComputation(V) // computes new P using latest V
      if 𝜋 == 𝜋' then return 𝜋, V
      𝜋 = 𝜋'
    }

Where:
* ValueIteration computes a transition matrix using a fixed policy, then iterates by recomputing values for each node using the previous values until either:
no value changes by more than the 'tol' flag,
or -iter iterations have taken place.
* GreedyPolicyComputation uses the current set of values to compute a new policy. If -min is not set, the policy is chosen to maximize rewards; if -min is set, the policy is chosen to minimize costs.

The value of an individual state is computed using the Bellman equation for a Markov property

    v(s) = r(s) + df * P * v

## Note
The program seems to not reach convergence when a discount factor that is too low is used.

Consulted with Curtis Yang for conceptual understanding.