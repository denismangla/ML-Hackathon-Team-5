"""Build the team PDF from experiment files and optional actual LAN demo screenshots."""
import csv
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans'})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

release=Path('results/release')
def read(path): return json.loads(Path(path).read_text())
def rows(path):
    with open(path) as f: return list(csv.DictReader(f))
def page(title, section):
    global fig,y,index
    index+=1
    fig=plt.figure(figsize=(8.27,11.69),facecolor='white')
    fig.text(.078,.952,'PATHSNAP   /   TEAM 05',size=9,weight='bold',color='#5545a5')
    fig.text(.925,.952,section,ha='right',size=8,color='#586375')
    fig.add_artist(Line2D([.078,.925],[.937,.937],transform=fig.transFigure,color='#cfd4df',lw=.8))
    fig.text(.075,.902,title,size=23,weight='bold',color='#18243c',va='top')
    fig.text(.078,.035,'PES UNIVERSITY   ·   UE24CS352A   ·   18 NOVEMBER 2026',size=8,color='#586375')
    fig.text(.925,.035,f'{index:02d}',ha='right',size=9,color='#5545a5')
    y=.833
def paragraph(s):
    global y
    lines=textwrap.wrap(s,89,break_long_words=False)
    fig.text(.079,y,'\n'.join(lines),va='top',size=10.6,linespacing=1.43,color='#23324a')
    y-=.0176*len(lines)+.019
def heading(s):
    global y
    fig.text(.079,y,s,va='top',size=13,weight='bold',color='#494086')
    y-=.032
def graphic(path,height):
    global y
    ax=fig.add_axes([.073,y-height,.854,height]);ax.imshow(plt.imread(path));ax.axis('off')
    y-=height+.013
def table(cols,data,widths=None,fs=9):
    global y
    height=.029*(len(data)+1)
    ax=fig.add_axes([.078,y-height,.845,height]);ax.axis('off')
    t=ax.table(cellText=data,colLabels=cols,colWidths=widths,cellLoc='left',colLoc='left',bbox=[0,0,1,1])
    t.auto_set_font_size(False);t.set_fontsize(fs)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('white');cell.set_facecolor('#e9e7f6' if r==0 else ['#f2f4f8','white'][r%2]);cell.PAD=.075
        cell.get_text().set_color('#302b54' if r==0 else '#243348')
        if r==0: cell.get_text().set_weight('bold')
    y-=height+.019
def save():
    if y<.04: raise ValueError(f'Page {index}: content exceeds safe area ({y})')
    pdf.savefig(fig);plt.close(fig)
def n(x):return f'{float(x):.3f}'
def pct(x):return f'{100*float(x):.2f}%'
def baseline_table(keys,cols):
    table(cols,[[names[m['model']]]+[n(m[k]) for k in keys] for m in models])
def explain(model,description,interpretation):
    m=next(row for row in models if row['model']==model)
    h=diag['models'][model]
    heading(names[model])
    paragraph(description)
    paragraph(f"Top-1 {pct(m['student_top1'])}, trip Macro F1 {n(m['student_macro_f1'])}, loss {n(m['student_log_loss'])}. Candidate-level balanced accuracy was {n(h['train_balanced_accuracy'])} on fit rows and {n(h['student_balanced_accuracy'])} on eligible student rows. "+interpretation)

def generate():
    page('From GPS points to a road sequence','Project and responsibilities')
    paragraph('A GPS reading can sit between parallel streets or near a junction. PathSnap ranks nearby directed arcs at each timestamp, then uses the sequence of observations to choose a plausible route through the road network.')
    heading('Experiment outcome')
    paragraph(f"The selected Random Forest achieved {pct(before['top1_accuracy'])} pointwise accuracy on 19,160 student-test observations. Viterbi produced {pct(after['top1_accuracy'])} accuracy and raised mean trip Macro F1 from {n(before['macro_f1'])} to {n(after['macro_f1'])}. The two metrics answer different questions: how often a point is correct and how evenly roads within a trip are recovered.")
    heading('Team ownership')
    table(['Member and SRN','Assigned experiment and responsibility'],[
        ['DIVYANSHI · CS161','Random Forest; data processing and features'],['DENIS MANGLA · CS151','Gaussian NB, Logistic Regression; decoder and API'],['G. YEKAMBAR ESHWAR RAO · CS173','HistGradientBoosting; tuning and shared UI'],['CH SOWMYA VAISHNAVI · CS126','KNN; shared UI, route map and report']], [.44,.56],8.5)
    paragraph('All SRNs begin PES2UG24. CS173 is GURUBELLI YEKAMBAR ESHWAR RAO. Each owner can reproduce their experiment using run_mine.py in their individual handoff.')
    heading('Evaluation scope')
    paragraph('The supplied train and student_test partitions were used for development. Reserved evaluator trips and a physical two-laptop run were unavailable during this verification. The deployment section records local test evidence and accepts both required live screenshots when rebuilt.')
    paragraph('AI assistance was used for implementation, analysis and drafting. The experiment scripts, saved model files, fold results and per-trip predictions accompany this report for team review and reproduction.')
    save()
    page('Dataset and candidate engineering','2  /  Exploratory data analysis')
    summary=read('data_stage1/summary.json')
    quality=rows('data_stage1/data_quality.csv');coverage=rows('data_stage1/candidate_coverage.csv')
    paragraph('Each trip provides road nodes (longitude, latitude), directed arcs (node indices), timestamped GPS coordinates and an ordered reference arc sequence. Validation checks finite numeric values, column counts, coordinate ranges, node and arc indices, and nondecreasing timestamps. All 360 files across 90 trips passed. No missing numeric fields required imputation.')
    table(['Partition','Trips','GPS points','Candidate rows','Reference misses'],[[part,str(len(rs)),f"{sum(int(r['observations']) for r in rs):,}",f"{sum(int(r['candidate_rows']) for r in rs):,}",pct(sum(int(r['candidate_misses']) for r in rs)/sum(int(r['observations']) for r in rs))] for part in ['train','student_test'] for rs in [[r for r in coverage if r['split']==part]]])
    graphic('data_stage1/dataset_overview.png',.208)
    paragraph(f"Across both partitions, {sum(int(r['duplicate_timestamps']) for r in quality):,} consecutive timestamps repeat. The largest gap is {max(float(r['max_gap_s']) for r in quality):,.0f} seconds. Long gaps break the sequence. Small or invalid movements suppress the heading feature.")
    heading('Candidate search and 14 features')
    paragraph('Project each trip into local metres about its median GPS position. Index short segment pieces in a KD-tree, then rank original arcs by exact point-to-segment distance. Keep up to eight within 100 m. If none are inside, retain the nearest candidates and flag the fallback.')
    paragraph('Distance to the segment, projection fraction, distances to both ends and segment length describe proximity. Heading difference, elapsed time, displacement, speed, gap and valid-heading flags describe motion. Fallback and endpoint degrees indicate sparse candidates or complex junctions.')
    save()
    page('Where the labels are uncertain','2  /  Reference alignment')
    paragraph('The supplied .route describes arc order. It gives no arc ID per GPS timestamp. A monotone alignment minimises squared distance along ordered reference occurrences with a small jump penalty. It restarts after gaps above 120 s. This creates the target used for supervised ranking and pointwise evaluation.')
    graphic('data_stage1/training_trip_map.png',.238)
    paragraph('Raw training trip 00000001. The GPS track and the supplied route are plotted in their original geographic coordinates. The training reference is used to derive targets separately from model inputs.')
    heading('Quality checks')
    paragraph(f"{summary['alignment_usable']:,} observations fall within 100 m of their aligned reference. Across both partitions, {summary['alignment_ambiguous']:,} have another distinct reference arc within 5 m of the chosen distance. The categories can overlap. Fit rows require a usable, unambiguous alignment and a reference among the candidates.")
    table(['Training trip','Points audited','Nearest ref., median','Aligned, median','Beyond 100 m'],[[r['trip'],r['sample'],f"{r['median_nearest_reference_m']:.1f} m",f"{r['median_aligned_m']:.1f} m",r['aligned_above_100m']] for r in diag['alignment']],fs=8.4)
    paragraph('Even an unconstrained nearest reference is distant on trips 54 and 33. Relaxing the ordering alone would therefore leave large residuals. Trip 11 has low median distance but contributes zero eligible fit rows. These spot checks use evenly spaced observations and do not establish ground truth.')
    save()
    page('Five models on the same split','3  /  Model comparison')
    paragraph('For each candidate row, the binary target indicates whether its arc matches the derived reference. The highest positive-class score wins each observation. All five models use identical sampled data and three whole-trip validation folds. Grouped CV Macro F1 gives less frequent roads a voice in model selection.')
    baseline_table(['cv_macro_f1','student_top1','student_macro_f1','student_direct_continuity'],['Model','CV F1','Test Top-1','Test F1','Continuity'])
    baseline_table(['student_log_loss','student_inference_seconds','fit_seconds'],['Model','Test loss','Score time (s)','Final fit (s)'])
    paragraph('Top-1 is pooled across observations. F1 is calculated per trip over the union of true and predicted arcs, then averaged. Continuity counts identical or directly adjacent arcs on eligible transitions. Scoring time is the total for ten trips with precomputed features. The pointwise probabilities are normalised within each observation; misses receive a reserved mass of 1e-12.')
    paragraph('Seed 42, up to 80 eligible observations per training trip, then a 20,000-row cap per fit. One of 80 training trips has no eligible rows. The three CV scores use complete validation trips, including ambiguous observations and misses. The ten student trips were held out from model selection.')
    save()
    page('Reading the model scores','3  /  Comparison plots')
    graphic('results/models/model_comparison.png',.29)
    paragraph('Grouped CV narrowly separates the two tree ensembles. Logistic regression tops the baseline student Top-1 column. Student results describe out-of-partition performance; the selection decision uses the training-fold scores.')
    graphic('results/models/loss_runtime.png',.26)
    paragraph('KNN spends most of its scoring time on neighbour lookup. Its large loss indicates that some held-out observations assigned the reference candidate very little probability. Both timing charts use identical student trips.')
    save()
    page('Linear, probabilistic and local models','3  /  Individual experiments')
    explain('logistic_regression','A scaled, L2-regularised logistic classifier provides a fast linear baseline. C=1 and max_iter=500. It minimises regularised binary log loss. It can weight distance and heading together, but an additive linear score cannot express all junction interactions.','The held-out value exceeds the fit value. This diagnostic gives little evidence of overfitting.')
    explain('gaussian_nb','Gaussian Naive Bayes fits class priors, per-class means and variances by likelihood, with variance smoothing of 1e-9 and scaled inputs. Training is inexpensive. Its conditional-independence assumption is difficult for correlated distances and motion features.','Both values are comparatively low. This is consistent with a restrictive decision boundary on these features.')
    explain('knn','KNN uses seven distance-weighted neighbours in standardised Euclidean space and a KD-tree. It stores labelled rows and votes locally, with no iterative loss optimisation. Similar geometries can share labels, but sparse regions and class overlap make probabilities unstable.','Perfect fit-row voting and a lower student value are consistent with memorisation. Its loss and scoring time were the highest of the five.')
    save()
    page('Tree ensembles and model selection','3  /  Individual experiments')
    explain('hist_gradient_boosting','Histogram gradient boosting fits successive trees to binary log loss, with 100 iterations, learning rate 0.1, 15 leaf nodes, L2=1 and early stopping disabled. It can model interactions between heading, offset and speed; small training samples constrain rare-road behaviour.','The candidate-level gap is small. It delivered the lowest baseline student log loss, although its top-ranked roads yielded lower student F1 than the forest.')
    explain('random_forest','The baseline forest uses 80 bootstrapped trees, depth 16 and minimum leaf size 3. Each split uses Gini impurity and the forest averages class probabilities. Nonlinear splits handle geometry interactions; deep trees can partition noisy labels too finely.','The fit-to-student gap supports trying stronger regularisation. Its grouped CV F1 was highest, narrowly ahead of boosting.')
    paragraph('Balanced accuracy here averages sensitivity and specificity at the classifier decision threshold, using the actual 20,000 fit rows and all eligible student candidate rows. It complements observation-level ranking. Every model used the same diagnostic mask; differences in trip mix can affect the gap.')
    save()
    page('Bayesian search for the selected forest','4  /  Hyperparameter optimisation')
    paragraph('Three seeded random configurations initialise a Gaussian process with a Matérn kernel. At each of seven further trials, the search picks the largest expected improvement among 512 proposed points. The objective is mean validation-trip Macro F1 across three grouped folds. The student partition is evaluated only after selection.')
    paragraph('Uniform draws on the unit cube map to 50–149 trees, depth 6–25 and minimum leaf size 1–9. The fixed seed is 42. Each trial refits the same capped training scheme.')
    table(['Trial','Proposal','Trees','Depth','Min. leaf','CV F1'],[[str(t['trial']+1),'Random' if t['trial']<3 else 'GP / EI',t['parameters']['n_estimators'],t['parameters']['max_depth'],t['parameters']['min_samples_leaf'],n(t['cv_macro_f1'])] for t in read('results/tuning/trials.json')],fs=8.5)
    table(['Forest','CV F1','Test Top-1','Test F1','Test loss','Score (s)'],[[label,n(tune[cv]),n(scores['top1_accuracy']),n(scores['macro_f1']),n(scores['log_loss']),n(scores['inference_seconds'])] for label,cv,scores in [('Baseline','untuned_cv_macro_f1',tune['untuned_student_test']),('Tuned','tuned_cv_macro_f1',tune['tuned_student_test'])]],fs=8.5)
    paragraph('The winning initial trial used 119 trees, depth 7 and leaf size 9. The seven later proposals did not exceed it. The shallower forest improved the selection objective and student F1, while student log loss increased. Candidate ranking and probability calibration can move in different directions.')
    save()
    page('Reconstructing a directed route','5  /  HMM and Viterbi')
    paragraph('Hidden states are candidate directed arcs at each timestamp. Emissions are normalised classifier probabilities. A transition scores the discrepancy between observed displacement and directed travel distance, including projection offsets and bounded Dijkstra distance between arc endpoints.')
    search=read('results/decoder/search.json')
    paragraph(f"Viterbi maximises the sum of log emissions and transition scores. A transition contributes minus absolute distance discrepancy divided by beta. With the selected forest configuration refit outside training fold 0, {len(search['validation_trips'])} held-out training trips gave mean Macro F1 "+', '.join(f"{n(v)} at {k} m" for k,v in search['mean_f1'].items())+f". The original scale was {search['selected_beta_m']:.0f} m.")
    paragraph('Search distance is capped at 3 km. Implausible travel beyond max(200 m, 60 m/s × elapsed seconds) is excluded. The decoder restarts at a gap over 120 s or when every transition is impossible. Backtracking recovers winning arcs; shortest-path intermediates fill connected sections without drawing across restarts.')
    seq=diag['sequence']
    table(['Student-test outcome','Classifier only','With Viterbi'],[[k,n(before[v]),n(after[v])] for k,v in [('Top-1 accuracy','top1_accuracy'),('Trip Macro F1','macro_f1'),('Direct continuity','direct_continuity')]]+[['Ordered sequence similarity',n(np.mean([s['classifier'] for s in seq])),n(np.mean([s['decoder'] for s in seq]))]], [.52,.24,.24])
    paragraph('Ordered similarity is the mean SequenceMatcher matching-block ratio between the supplied chronological .route and the observed arc sequence after collapsing adjacent repeats on both sides. This diagnostic uses selected observation states and excludes inserted bridge arcs. Direct continuity can penalise a legitimate multi-edge movement.')
    pairs=list(zip(dec['per_trip_before'],dec['per_trip_after']))
    lo=min(pairs,key=lambda p:p[1]['top1_accuracy']-p[0]['top1_accuracy'])
    paragraph(f"The largest pointwise decline was trip {lo[0]['trip_id']}: {pct(lo[0]['top1_accuracy'])} to {pct(lo[1]['top1_accuracy'])}. A consistent path can favour a neighbouring directed arc at ambiguous points. Decoder log loss is not reported because Viterbi outputs a best sequence, not marginal posterior probabilities.")
    save()
    if (release/'decoder_refinement.png').exists():
        page('Reducing GPS jitter errors','5  /  Decoder refinement')
        refinement = read('results/decoder/refinement_search.json')
        initial = read('results/decoder/baseline_summary.json')['decoder']
        chosen = refinement['chosen']['settings']
        paragraph(f"The refined decoder accepts a backward movement of at most {chosen['reverse_tolerance_m']:.0f} m on the same directed arc as GPS jitter. This avoids forcing a network loop when the observation only moved slightly backwards. A second search on the training validation trips selected an {chosen['beta_m']:.0f} m transition scale.")
        graphic(release/'decoder_refinement.png',.38)
        table(['Student-test outcome','Original decoder','Refined decoder'], [
            ('Point Top-1',pct(initial['top1_accuracy']),pct(after['top1_accuracy'])),
            ('Mean trip Macro F1',n(initial['macro_f1']),n(after['macro_f1'])),
            ('Macro precision',n(initial['macro_precision']),n(after['macro_precision'])),
            ('Macro recall',n(initial['macro_recall']),n(after['macro_recall'])),
            ('Direct continuity',pct(initial['direct_continuity']),pct(after['direct_continuity'])),
        ], [.46,.27,.27])
        paragraph('The same ten student-test trips and derived reference labels were used for both versions. The reserved evaluator trips were unavailable.')
        save()
    page('Compare the four geographic layers','6  /  Folium map')
    graphic(release/'folium_layers.png',.395)
    paragraph('Student-test trip 00000009. This static export reads the road lines, reference lines, ordered predicted sections and GPS marker coordinates directly from the generated Folium layer objects. Axes show geographic coordinates. The interactive HTML adds a basemap, zoom controls and layer selection.')
    s=read(release/'sample_response.json')
    paragraph(f"The {len(s['predicted_arc_ids']):,} observations produced {len(s['route_arc_sections'])} sections with {s['bridge_count']} connecting arcs. The dashed orange line is the reference and the solid teal line is the refined route. Toggle GPS or pointwise selection to inspect disagreement.")
    paragraph('Open results/release/PathSnap_Folium.html in a browser. Its extent is fitted to the GPS track. OSM tiles and Leaflet assets use an internet connection. In the evaluator dashboard the ordered route and GPS points are drawn locally from the API response, with no reference layer sent to the client.')
    save()
    page('Deploy and verify on two laptops','7  /  API and evaluator UI')
    sweep=read(release/'api_sweep.json')
    paragraph(f"All {len(sweep)} student-test trip requests returned HTTP 200 in an in-process FastAPI test. Across {sum(r['observations'] for r in sweep):,} observations, decoded IDs matched the saved batch predictions and every neighbouring pair of displayed route arcs shared a directed endpoint. These checks cover server inference and geometry. Browser rendering and LAN reachability require separate live evidence.")
    paragraph('The React server-render check verified the sample GPS markers, route and network lines, and all five model report entries. Client interactions still need the live browser demonstration.')
    heading('Laptop 1 · Host the model and API')
    paragraph('From the project directory run python demo.py server --data "<partition directory>". Add --mode evaluator for the reserved trips. The partition contains eight-digit trip folders. The server loads the tuned model, extracts features and decodes. It then scores against the server-side reference when present. GET /health, GET /trips and POST /predict serve the evaluator.')
    heading('Laptop 2 · Run the interface')
    paragraph('Run python demo.py ui and open http://127.0.0.1:8090. Enter Laptop 1’s LAN address with port 8000. Connect, choose one of the returned trips and select Reconstruct route. Confirm the GPS observations, Top-1, Macro F1, time and route sections. Toggle GPS and roads, zoom, pan and fit to inspect a junction.')
    evidence=[('ui_demo.png','Evaluator UI on Laptop 2'),('receiving_laptop.png','Successful request and response summary on Laptop 1')]
    heading('Live deployment evidence')
    for filename,label in evidence:
        paragraph(f"{label} — "+('image included on the following evidence pages.' if (release/filename).is_file() else f'awaiting physical capture. Save as results/release/{filename}.'))
    paragraph('Rebuild with python -m scripts.build_report after the live run. Record the same trip ID in both captures. Publish the team-reviewed repository and submit its public URL and the refreshed technical PDF through the organiser’s form.')
    save()
    for filename,label in evidence:
        if (release/filename).is_file():
            page('Live demonstration evidence','7  /  Deployment')
            heading(label);graphic(release/filename,.63);save()

if __name__=='__main__':
    index=0
    models=rows('results/models/comparison.csv')
    names={'logistic_regression':'Logistic regression','gaussian_nb':'Gaussian NB','hist_gradient_boosting':'HistGradientBoosting','random_forest':'Random forest','knn':'KNN'}
    diag=read(release/'diagnostics.json');tune=read('results/tuning/summary.json')
    dec=read('results/decoder/summary.json');before=dec['classifier'];after=dec['decoder']
    Path('docs').mkdir(exist_ok=True)
    with PdfPages('docs/PathSnap_Team5_Technical_Report.pdf') as pdf:
        generate()
