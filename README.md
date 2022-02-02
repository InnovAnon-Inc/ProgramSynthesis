# ProgramSynthesis
Giving the Holy Grail another college try. The program(s) and its set of outputs are public domain.

The InnovAnon algorithm is described as a program that will ultimately produce all possible programs.
According to the mythos, the algorithm is run in an (acausal) environment s.t. the non-termination of the program is not relevant:
all outputs are considered to already exist.

In our causal universe, implementations should take run-time into consideration,
prioritizing to output interesting/useful programs first--whatever that means.

Our first implementation was in Java; it was inelegant, as it generated syntactical strings, rather than directly leveraging ASTs.
It was a one-shot program generator, and so iterating any subset of the search space would have been unbounded.
After seeing the code and its output, Dr. Shonle hired me to implement the first version of La Clase Magica.

Our second implementation was also in Java. We called it Simon, the Drawer-er. It instantiated random objects.
This one produced results that we found interesting. Specifically, it could instantiate Java Canvas objects and draw on them something akin to artwork.
It could be used to iterate a search space. It was used in the industry (at Trinity Millennium Group) for data augmentation and advanced mocking.

Our third implementation was in Python, and had a greater degree of introspection, as it not only tracks its recursion depth,
but it also vectorizes its call stack.
The call stack vector is used as input to a neural network to decide the order in which to iterate the search space.
A vanilla feed-forward NN did not produce interesting results.
This version was a one-shot program generator.

This is the fourth implementation. The goal is to iterate the search space.
Once this feature is functional, we will begin integrating AI/ML.
The proposed architecture is a recursive cellular automata:
to evolve memory-augmented cellular neural networks using genetic cellular automata.

