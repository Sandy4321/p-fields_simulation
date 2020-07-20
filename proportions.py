import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.spatial import KDTree

def prop_reals(reals, cats):
  props = {}
  for c in cats:
    cat_props = []
    for r in reals:
      cat_filter = np.array(r) == c
      real_prop = len(np.array(r)[cat_filter])/len(r)
      cat_props.append(real_prop)
    props[c] = cat_props
  return props

def cat_plot(target, reals, weights=None, title='Proportions', savefig=None):
  #plotting target declustered histogram
  weights = np.array(weights) if weights is not None else np.array([1/len(target)] * len(target))
  cats = np.unique(target)
  cats_labels = [i for i in range(len(cats))]
  cat_dict = {}
  for c in cats:
    mask = target == c
    height = weights[mask].sum()
    cat_dict[c] = height
  plt.bar(cats_labels, cat_dict.values())
  plt.ylabel('Proportion')
  plt.xlabel('Categories')
  plt.title(title)
  plt.xticks(cats_labels, [str(int(i)) for i in cat_dict.keys()])
  #plotting realizations boxplots
  reals_props = prop_reals(reals, cats)
  plt.boxplot(reals_props.values(), positions=cats_labels, manage_ticks=False)
  if savefig is not None:
    plt.savefig(savefig, dpi=500)
  plt.show()

def closest_node(grid, x, y, z):
  coords = grid.locations()
  kdtree = KDTree(coords)
  points = ((i, j, k) for i, j, k in zip(x,y,z))
  idxs = []
  for p in points:
        dist, neighs = kdtree.query(p, k=1)
        idxs.append(neighs)
  return idxs

def back_flag(grid, reals, x, y, z, values, savefig=None):
  if z is None:
        z = np.zeros(len(x))
  codes = np.unique(values)
  ids = closest_node(grid, x, y, z) 
  reals_values = [np.array(r)[ids] for r in reals]
  cms = [confusion_matrix(values, pred) for pred in reals_values]
  sum_ew = np.sum(cms, axis=0)
  final_cm = sum_ew / sum_ew.astype(np.float).sum(axis=1)
  plt.figure()
  sns_plot = sns.heatmap(final_cm, annot=True, vmin=0.0, vmax=1.0, fmt='.2f')
  plt.yticks(np.arange(len(codes))+0.5, labels=codes)
  plt.xticks(np.arange(len(codes))+0.5, labels=codes)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  if savefig is not None:
    plt.savefig(savefig, dpi=500)
  figure = sns_plot.get_figure()