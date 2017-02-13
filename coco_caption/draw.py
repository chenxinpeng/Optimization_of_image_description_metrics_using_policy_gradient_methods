#! encoding: UTF-8

import cPickle as pickle
import matplotlib.pyplot as plt

with open("Bleu_1.pkl", "r") as f:
    Bleu_1 = pickle.load(f)

with open("Bleu_2.pkl", "r") as f:
    Bleu_2 = pickle.load(f)

with open("Bleu_3.pkl", "r") as f:
    Bleu_3 = pickle.load(f)

with open("Bleu_4.pkl", "r") as f:
    Bleu_4 = pickle.load(f)

with open("METEOR.pkl", "r") as f:
    METEOR = pickle.load(f)

with open("CIDEr.pkl", "r") as f:
    CIDEr = pickle.load(f)

print len(Bleu_1)

plt.plot(range(0, 2*len(Bleu_1), 2), Bleu_1, label="Bleu-1", color="g")
plt.plot(range(0, 2*len(Bleu_2), 2), Bleu_2, label="Bleu-2", color="r")
plt.plot(range(0, 2*len(Bleu_3), 2), Bleu_3, label="Bleu-3", color="b")
plt.plot(range(0, 2*len(Bleu_4), 2), Bleu_4, label="Bleu-4", color="m")
plt.plot(range(0, 2*len(METEOR), 2), METEOR, label="METEOR", color="k")
plt.plot(range(0, 2*len(CIDEr), 2), CIDEr, label="CIDEr", color="y")

plt.grid(True)
#plt.legend(handles=[line_1, line_2])
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.legend(handles=[line1], loc=1)
plt.legend(loc=2)
plt.show()
#plt.savefig("tmp.png")
