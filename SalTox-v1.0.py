import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from tkinter import filedialog



data = pd.read_excel("lib\\train.xlsx", index_col=0)
xtr = data.iloc[:,:-1]
ytr = data.iloc[:,-1]

reg = PLSRegression(n_components=3).fit(xtr, ytr)
filepath = filedialog.askopenfilename(initialdir="")

data2 = pd.read_excel(filepath, index_col=0)

xte = data2[xtr.columns.tolist()].copy()

tep = pd.DataFrame(reg.predict(xte), columns=["Predictions"], index=xte.index.values)
tep.to_excel("Predictions.xlsx", index_label="IDs")