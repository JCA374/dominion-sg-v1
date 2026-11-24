# Episode 01: A ratio that was wrong from the first step

This series follows one question. Can a slow search teach our best Dominion network something lasting? Before any of that could run, I had to fix the training itself. This first episode is about one small piece of mathematics, and why a tiny mismatch in it matters.

Our network is trained with an algorithm called proximal policy optimisation. People say P P O. Here is the idea in words. The network plays three thousand games. At every decision it does not simply take its favourite option. It draws an option at random, with a probability for each one. Then, after the games, it asks a question about every decision it made. Did that choice turn out better or worse than expected? Better choices get a higher probability. Worse ones get a lower probability.

The key quantity is a ratio. Take the probability that the new network gives to the move that was played. Divide it by the probability that the old network gave to the same move, at the moment it was played. If the network has not changed, that ratio is exactly one. P P O multiplies each lesson by this ratio. And it clips it. If the ratio wanders outside a band from point eight to one point two, the gradient through that decision stops in one direction. The clip is a safety rail. It keeps each update small.

Now, the probabilities. The network outputs a score for each option, called a logit. To turn scores into probabilities you exponentiate them and divide by their sum. That is the softmax. There is one extra knob, called the temperature. Before the softmax, you divide every logit by the temperature. A temperature below one sharpens the distribution: the favourite gets more likely. A temperature above one flattens it.

Here is the bug. The game engine, written in C, drew moves using the temperature. It recorded the log probability of the move it drew, with the temperature included. But the training code, in PyTorch, computed the new probability without the temperature. So at the very first step, before anything had changed, the two probabilities came from two different softmaxes. The ratio was not one.

How big was the error? The reviewer measured it with our best network's real weights, at a temperature of point seven, on nine hundred and sixty three real decisions. The ratio was off by about five percent on average. And about three and a half percent of all decisions were already outside the clipping band, before a single update. For those decisions the safety rail was engaged for no reason at all.

Why was it never noticed? Because our runs used a temperature of exactly one. Dividing by one changes nothing. The bug only appears at other temperatures. That is a lesson worth keeping. A check that runs only at the default setting cannot find a bug that only lives at the other settings.

The fix is one line. The update now divides the logits by the same temperature as the sampler, and the entropy term uses the same distribution. Then I added a test that plays real games with the real network at temperature point seven and at temperature one. It compares the probability the engine recorded with the probability PyTorch computes. After the fix, the average error is about three parts in ten million. That is just float rounding. Every training run now does this check at step zero, and refuses to start if it fails.

There were other fixes of the same kind. Games the engine reports as broken no longer feed the training. A game whose record buffer overflowed is dropped too, because a truncated game gives wrong credit to the moves before the cut. The random stream that chooses kingdoms and opponents used to depend on the run's name. Now it depends on a seed. And the results come back from the worker processes in a fixed order. Together these mean that two runs with the same seed write identical weights, bit for bit. That matters for everything that follows. When we compare two training branches, the only difference between them must be the one we put there on purpose.

Next episode: how the teacher decides that one move is better than another, and why picking the best of six noisy numbers fools you.
