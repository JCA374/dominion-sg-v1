# Episode 03: Six hundred games, four arms, one build

You might remember earlier numbers for the teacher: fifty nine percent with buy search, and sixty two with buys and actions together. They came from different games, different builds and different days. The handoff asked for something cleaner. All arms on the same games, with the same code, before we build anything on top of them. This episode is about that experiment, and about how to read its intervals.

First, the design. I wrote a frozen list of three hundred fresh kingdoms, each with a seed, before any game was played. None of them appears in any earlier test, and none will ever be used for training. Each kingdom is played twice, once with our teacher in each seat. So every arm plays six hundred games, and all four arms play exactly the same six hundred. I also froze one compiled copy of the teacher, and every match records a fingerprint of that file and of every source file. If anyone resumes a match with different code, it refuses.

The four arms. A mirror: our network against itself, with no search. Then the buy teacher. Then the action teacher, which searches when to play which action card and when to stop. And both together.

The mirror is a nice check. Our network plays its single favourite move, with no randomness. If both seats are the same network on the same seed, the game played from seat one and the game played from seat two are the same game. One copy wins exactly when the other loses. So every kingdom gives exactly one half, and the mirror must score exactly fifty percent, with an interval of zero width. It did. That tells you the pairing is right.

Now the unit of evidence. A single game is not independent of its partner, because the two seats share the kingdom. So the unit is a kingdom pair: the average of the two games. That average can be zero, one quarter, one half, three quarters or one. The score is the mean over three hundred pairs. For the interval I resample the three hundred pairs with replacement four thousand times, recompute the mean each time, and take the middle ninety five percent. That is a bootstrap. It needs no assumption about the shape of the distribution.

The results. The buy teacher scored fifty six percent, with an interval from fifty three to fifty nine. The action teacher scored fifty five point one, from about fifty two and a half to fifty eight. Both together scored fifty eight point eight, from about fifty five and a half to sixty two. Not a single game had an engine error. Between one and four games per arm reached the round limit.

Then the paired differences, which are the real question. Because all arms played the same kingdoms, I can subtract them pair by pair. That removes the variation between kingdoms, just as pairing removed the variation between deals last episode. Buy search minus nothing: six points, from three to nine. Action search minus nothing: about five points. Both minus buy alone: two point eight points, but the interval runs from minus one point two to plus six point seven. So adding action search on top of buy search may help, but this test cannot show it yet.

Speed matters too. The buy teacher's median decision took about point eight seconds, and one in twenty took over two and a half seconds, on a shared laptop. That is far too slow to play like that for real. That is the reason for the whole project: to move what the search knows into the network, where it costs nothing.

The rule for what to label next was written before the results. Take the simplest arm that clearly beats plain play. The buy teacher's interval sits above fifty percent, so the pilot labelled buys only. Next episode: how to teach our network those buys without breaking everything else it knows.
