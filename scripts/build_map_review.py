"""Create the review PDF from measured experiment and geometry outputs."""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT=Path('results/figures')
folder=Path('results/decoder')
before=json.loads((folder/'baseline_summary.json').read_text())['decoder']
after=json.loads((folder/'summary.json').read_text())['decoder']
rows=[
    ('GPS trajectory and candidate zoom',OUT/'GPS_Trajectory_and_Candidate_Zoom.png',
    'Trip 00000009. The left plot follows 1,049 GPS observations in time order. The right plot shows the supplied reference, candidate roads and the road selected by the decoder at GPS observation 300. Coordinates are metres east and north of the trip centre.'),
    ('Dataset geometry and sampling',OUT/'Dataset_Geometry_and_Sampling.png',
     'This covers all 80 training and 10 student-test trips. Road-network size counts directed arcs. Extent is the bounding-box diagonal. A GPS jump counts a step with positive elapsed time and implied speed above 60 m/s.'),
    ('Candidate convergence',OUT/'Candidate_Convergence.png',
     'The left plot varies radius at K=20. The right plot varies K at a 100 m radius. Both count whether the derived reference road is present in the candidate set. This diagnostic uses strict radius bounds.'),
    ('Decoder accuracy and route quality',Path('results/release/decoder_refinement.png'),
     f"On the same {after['observations']:,} student-test GPS points, refined Viterbi raised Top-1 from {before['top1_accuracy']:.2%} to {after['top1_accuracy']:.2%}, trip Macro F1 from {before['macro_f1']:.4f} to {after['macro_f1']:.4f}, and direct continuity from {before['direct_continuity']:.2%} to {after['direct_continuity']:.2%}. Settings were chosen on training validation trips."),
]
with PdfPages(OUT/'PathSnap_Maps_and_Metrics.pdf') as pdf:
    for index,(title,image,note) in enumerate(rows,1):
        fig=plt.figure(figsize=(11.69,8.27),facecolor='#fbfbf6')
        fig.text(.075,.93,'PATHSNAP   /   TEAM 5   /   GEOGRAPHIC EVIDENCE',color='#13766a',weight='bold',size=11)
        fig.text(.075,.855,title,color='#182f2b',weight='bold',size=23)
        ax=fig.add_axes([.075,.18,.85,.635]);ax.imshow(plt.imread(image));ax.axis('off')
        fig.text(.075,.14,'\n'.join(textwrap.wrap(note,135)),va='top',color='#38544d',size=10.5,linespacing=1.45)
        fig.text(.925,.04,f'{index} / {len(rows)}',ha='right',size=9,color='#38544d')
        pdf.savefig(fig);plt.close(fig)
