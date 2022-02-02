@dataclass
class  ML(object):
    """ Machine Learning parent class """
    pass

@dataclass
class  RF(ML):
    """ Random Forest """
    pass
@dataclass
class KNN(ML):
    """ K-Nearest Neighbor """
    pass
@dataclass
class SVM(ML):
    """ Support Vector Machine """
    pass

@dataclass
class  GA(ML):
    """ Genetic Algorithm """
    pass
@dataclass
class CGA(CA,ML):
    """ Cellular Genetic Algorithm """
    pass

@dataclass
class  NN(ML):
    """ Neural Network parent class """
    pass

@dataclass
class FNN(NN):
    """ Fully-Connected Feed-Forward Neural Network """
    # TODO with drop-out
    pass
@dataclass
class CNN(NN):
    """ Convolutional Neural Network """
    # TODO with drop-out
    pass
@dataclass
class RNN(NN):
    """ Recurrent Neural Network """
    # TODO with drop-out
    pass
#@dataclass
#class rNN(NN):
#    """ Recursive Neural Network """
#    pass
@dataclass
class GNN(NN):
    """ Graph Neural Network """
    # TODO with drop-out
    pass

@dataclass
class MAML:
    """ Memory-Augmented Machine Learning """
    nn: ML

@dataclass
class  ED(NN):
    """ Encoder-Decoder Network """
    enc: ML
    dec: ML

#@dataclass
#class GAN:
#    """ Generative Adversarial Network """
#    gen: ML
#    dis: ML

@dataclass
class MGAN:
    """ Multi-Generative Adversarial Network """
    gen: Iterable[ML]
    dis: Iterable[ML]

#@dataclass
#class Gym:
#    sm  : Callable[[OV], IV]
#    util: Callable[[OV], float]

@dataclass
class AGI:
    """ Multi-Task Machine Learning Algorithm """
    ai : ML # recursive GCA: each type of ML is recursive
    gym: Iterable[Gym] # memory of problems & on-going problems

    # needs to continually enhance accuracy and speed on problems, including:
    # - i/o
    # - reproduction, self-optimization
    # - generating more problems

    # need a way for cell(s) to destroy another (i.e., anti-cancer)

    def train(self):
        ai2 = map(ai.train, self.gym)



# raw_input
# pynput
# file operations
# TODO network
# opencv
# sounddevice









class Cell:
    # each cell is a thread
    inp: list[object]
    out: list[object]
    io : list[object] # 2-way
    f  : Callable[[ov], iv]
class State: # graph 
    #dim      : list[int]
    cells    : list[Cell]
    neighbors: Callable[[Cell], list[Cell]] # padded with i/o devices



class CA:
    init  : Callable[[State], ] # initialize to a state where cells can do basic functionality
    update: Callable[[State], State] # cross-over using an AST
    
    def run():
        while True:
            # converge()
            # TODO increase mutation rate ?
            #      add new cell types ?

            # need growth and pruning phases

            # need a way to send new update functions
            pass 





    


#class A:
#    rule : Callable[[ACell, list[ACell]], ACell]

class  V:
    ms : object # memory space
    hd : object # hard drive(s)
    nc : object # network card
class IV(V):
    """ Input Vector """
    pass
class OV(V):
    """ Output Vector """
    pass
class MTCell(GCell):
    def update(iv):
        return ov
    pass

class MTGCA:
    """ Multi-Task Genetic Cellular Automata """
    pass

    


# utility function:
# - expand capabilities and optimize architecture
#   (solve more problems, more accurately, more quickly)
# - accelerate evolution and ubiquity of self and others?
# - maximize chance that human should not want to press the stop-button?

# access to various I/O devices

# multi-gan: decide own opponent?

#task: evolve a map of (ip:port) connections using a variety of fitness functions
#      each cell is a server/thread and has various I/O devices... qemu instance
#task: generate fitness functions
#task: generate gyms

#task: reproduce (quine?), repair?, self-optimize
#task: survive reboot
#task: share interesting knowledge / novel representations with other sentients

# drop-out => can accept arbitrary i/o dims
# how to augment own dataset? how to reflectively learn so as to never converge?






# python GPS
# python cam
# python mic
# text  input
# mouse input
# battery level
# network card
# arbitrary input devices?
# - neurological electrodes

# graphic output
# speaker output
# text output

# file system access
# internet access
# financial access



# TODO
# - self-aware: own source code & state as input
# - all-knowing: game tree as input
# - aware of position in space
# - aware of temperature, pressure, humidity
# - alive:
#   - eat (recharge)
#   - shit?
#   - reproduce (build more machines)

# evolutionary pressure: GAN?
# how to: incremental learning?

# how to unbounded enhancement

# transfer learning
# multi-domain learning
# self-supervised learning



# lnn: liquid/dynamic state
# committee/associative
# cascading
# ptr-nn: pointer memory

# multi-task learning
# drop-out

# ptr-nn + encoder-decoder => text to code ?

# memory-augmented gnn with drop-out encoder-decoder multi-discriminator gan


