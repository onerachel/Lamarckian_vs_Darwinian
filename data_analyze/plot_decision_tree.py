import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import dtreeviz


# Prepare the data data
path = "/Users/lj/revolve2-Alife"
df = pd.read_csv(path+"/databases_eval1000/robotzoo_descriptors_fitness.csv")

X=df.iloc[:,6:-7]
y=df['label']

# Fit the classifier with default hyper-parameters
clf = DecisionTreeClassifier(max_depth=3)
clf_model = clf.fit(X, y)


## Plot1
# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(clf,
#                    feature_names=X.columns,
#                    class_names=df['avg_avg_framework'].unique(),
#                    filled=True)
# fig.show()
# fig.savefig(path+"/databases_eval580/plot_images/decision_tree.png")

# Plot2
viz_model = dtreeviz.model(clf,
                           X_train=X, y_train=y,
                           feature_names=X.columns,
                           target_name='avg_avg_framework',
                           class_names=df['avg_avg_framework'].unique())


fig = viz_model.view()     # render as SVG into internal object
fig.save(path+"/databases_eval1000/plot_images/decision_tree_advanced.svg")