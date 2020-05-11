import numpy as np
import matplotlib.pyplot as plt

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

def cat_plot(target, reals, weights=None):
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
  plt.xticks(cats_labels, [str(i) for i in cat_dict.keys()])
  #plotting realizations boxplots
  reals_props = prop_reals(reals, cats)
  plt.boxplot(reals_props.values(), positions=cats_labels, manage_ticks=False)
  plt.show()