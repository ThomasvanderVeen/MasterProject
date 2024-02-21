from functions import *

tp, tn, fp, fn = 6, 3, 1, 2
MCC = matthews_correlation(tp, tn, fp, fn)

print(MCC)