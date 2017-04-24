import Reinforcement

a = Reinforcement.ReinforcementLearning(lrate=1, trainfn='trainlr1', testfn='testlr1')
a.learn()
a.test()