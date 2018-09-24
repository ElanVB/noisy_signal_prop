import numpy as np
import seaborn as sns

def get_colours(n_colours, n_shades, reverse=False):
    # get base colours
    colours = ["blue", "green", "amber", "red", "purple", "brown", "grey"]
    base_colors = sns.xkcd_palette(colours)

    # get shades for each colour
    shades = np.array(
        [sns.light_palette(base, n_shades + 1, reverse=reverse)[1:] for base in base_colors]
    )
    return shades