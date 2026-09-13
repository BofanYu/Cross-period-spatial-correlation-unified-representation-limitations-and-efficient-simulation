"""Figures derived directly from fitted correlation curves."""
import matplotlib.pyplot as plt
from scripts.common import plotting as p

def move_panel_labels(fig,n):
    for ax in fig.axes[:n]:
        title=ax.get_title()
        ax.set_title("")
        ax.text(.5,-.215,title,transform=ax.transAxes,ha="center",va="top",fontsize=21,clip_on=False)
    fig.subplots_adjust(bottom=.12,hspace=.37)
