#from tsai.models.PatchTST import PatchTST
#help(PatchTST)
import json, matplotlib.pyplot as plt

with open("reports/loss_history.json") as f:
    hist = json.load(f)

plt.plot(hist["stockformer"]["train"], label="Stockformer Train")
plt.plot(hist["stockformer"]["val"], label="Stockformer Val")
plt.plot(hist["patchtst"]["train"], label="PatchTST Train")
plt.plot(hist["patchtst"]["val"], label="PatchTST Val")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss");
plt.show()
