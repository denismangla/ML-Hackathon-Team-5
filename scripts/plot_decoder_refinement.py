"""Compare decoder versions on the same student-test observations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    results = Path('results/decoder')
    original = json.loads((results/'baseline_summary.json').read_text())['decoder']
    latest = json.loads((results/'summary.json').read_text())
    columns = ['top1_accuracy','macro_f1','macro_precision','macro_recall','direct_continuity']
    labels = ['Top-1','Trip Macro F1','Macro precision','Macro recall','Continuity']
    stages = [
        ('Pointwise',latest['classifier'],'#929d9a'),
        ('Original Viterbi',original,'#d38a43'),
        ('Refined Viterbi',latest['decoder'],'#08766a'),
    ]
    fig,ax = plt.subplots(figsize=(10.6,4.8))
    ticks = np.arange(len(columns))
    for offset,(name,values,color) in zip((-.24,0,.24),stages):
        bars = ax.bar(ticks+offset,[100*values[k] for k in columns],width=.22,color=color,label=name)
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    ax.set(xticks=ticks,xticklabels=labels,ylim=(0,115),ylabel='Score (%)',title='Route decoding on 19,160 held-out student GPS observations')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center',ncol=3,frameon=False)
    fig.tight_layout()
    output = Path('results/release/decoder_refinement.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,dpi=190)
    plt.close(fig)


if __name__=='__main__':
    main()
