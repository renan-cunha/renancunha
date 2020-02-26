+++
title = "Brainstorming Benchmarks for Contextual Bandits"
date = "2020-02-25"
categories = ["AI", "Reinforcement Learning", "Contextual Bandits"]
+++

Since contextual bandits are relatively new, they haven't been studied and tested
so heavily as standard multi-armed bandits algorithms [1]. This being 
the case, it's useful to design benchmarks for contextual bandits so these
algorithms can be reliably tested, compared and improved.

## Prerequisites

* [**_A Very Short Intro to Contextual Bandits_**](/posts/contextual-bandits/)

## Simplest benchmark

[**_Many_**](https://twitter.com/pcastr/status/1226661835971145731) 
[**_people_**](https://twitter.com/karpathy/status/1013244313327681536) in the 
AI community [**_recommend_**](https://rockt.github.io/2018/08/29/msc-advice) 
the use of toy problems, these low complexity examples 
provide better and faster 
[**_debugging cycles_**](https://ai.stanford.edu/~zayd/why-is-machine-learning-hard.html),
a good property of early-stage research and experimentation. So now, let's try to imagine the simplest benchmark for the contextual bandits
problem. First, let's
use binary everywhere. 
Second, let's make the environment deterministic (an action *k* with context *x* 
always returns the same reward *r*). Third and finally, let's make
the reward function *f(x, k*) as simple as possible. Remember that this reward
function is calculated given the context observed and the action chosen.

So, this is what I came about:

* Context: 1 binary feature
* Number of actions: 2 
* Reward function pseudocode:

```
if context == 0
    action 0 gives a reward of 1, other actions give 0 
    rewards
else
    action 1 gives a reward of 1, other actions give 0 
    rewards
```

In this case, the best policy is to choose the $0^{th}$ action if the context is 0
and choose the $1^{st}$ action otherwise, easy.

## How to complicate

Now, let's try to incrementally add complexity to this problem.

### Stochastic rewards

Stochasticity always makes things harder and since this is what we are looking 
for, let's add it to our environment.

* Context: 1 binary feature
* Number of actions: 2
* Reward function pseudocode:

```
if context == 0
    action 0 returns a reward of 1 with 50% chance, other 
    actions give 0 rewards
else
    action 1 returns a reward of 1 with 50% chance, other 
    actions give 0 rewards
```

Here, the best policy it's still the same from the previous environment, but,
intuitively, it should be harder for an algorithm to discover it. 

### Continuous context

Instead of the context being made of only 1 binary feature, it makes sense to use
continuous ones, in case of the age of someone, for example, is being represented
as a feature. So, let's change 
the context possibilities to infinity!

* Context: 1 continuous feature on the interval [0, 1]
* Number of actions: 2
* Reward function pseudocode:

```
if context > 0.5
    action 0 gives a reward of 1 with 50% chance, other 
    actions give 0 rewards
else
    action 1 gives a reward of 1 with 50% chance, other 
    actions give 0 rewards
```

Note that we have to change the first ```if``` case of our reward function to
deal with the continuous feature, instead of testing equality, we test if the
context is greater than a threshold, in this case, 0.5. 

### Other ways

There are of course other ways to increase the complexity of the problem,
I will list some below.

### Datasets

While it is good to manually tweak a contextual bandit environment for research
or some experiments, 
this is not a straight forward task and requires dedicated human thought. A 
good alternative is to test contextual
bandit algorithms on supervised-learning datasets. This can be done
by representing the context and action by each (*X, y*) feature vector pair 
of the dataset, both regression
and classification datasets can be used. With this approach, the environment can randomly sample
one instance of the dataset, and the reward is based on the $y^k$ raw value in case
the of regression. With classification datasets, we can use accuracy as the reward.

## Some possibilities for benchmarks

Here, I try to imagine more benchmarks for this problem, trying to add complexity
and deal with corner cases.

* Standard multi-armed bandits scenario: the context doesn't influence the 
  reward function.
* Imbalanced Dataset: There is an action (or subsect of actions) that are the
  better most often.
* Type of reward function function instead of just ```context > 0.5```: 
    * linear function
    * non-linear function (neural network)
* Many degrees of randomness: This can be made by artificially adding noise to
  the dataset.
* Vary the number of actions and the number of context features: It would be good
  to test algorithms on datasets that have a high and low number of actions, this can
  also be said about the dimensionality of the context.
* Vary the number of samples: If the context repeats more often, then it
  should be easier for the algorithm to learn.

## References

* [1] Cortes, David. “Adapting Multi-Armed Bandits Policies to Contextual Bandits Scenarios.” ArXiv:1811.04383 [Cs, Stat], Nov. 2019. arXiv.org, http://arxiv.org/abs/1811.04383. 
* [2] Bietti, Alberto, et al. “A Contextual Bandit Bake-Off.” ArXiv:1802.04064 [Cs, Stat], Jan. 2020. arXiv.org, http://arxiv.org/abs/1802.04064.

Next Article: [**_Solving Contextual Bandits with Greediness_**](/posts/greedy/)
Previous Article: [**_A Very Short Intro to Contextual Bandits_**](/posts/contextual-bandits/)
