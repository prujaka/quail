from p1 import *

Restart = {
    "File" : "p1_final.pkl",
    "StartFromFileTime" : True
}

TimeStepping.update({
    "EndTime" : 54.,
    "num_time_steps" : 1500,
})

Numerics["InterpOrder"] = 2
Output["Prefix"] = "p2"
Output["AutoProcess"] = False
