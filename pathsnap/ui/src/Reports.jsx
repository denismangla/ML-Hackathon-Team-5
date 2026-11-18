import React,{useState} from 'react';

const fixed=(value,digits=4)=>value==null?'—':Number(value).toFixed(digits);
const percent=value=>fixed(value*100,2)+'%';
const metrics=[['Top-1 accuracy','top1_accuracy',percent],['Trip Macro F1','macro_f1',fixed],
  ['Macro precision','macro_precision',fixed],['Macro recall','macro_recall',fixed],
  ['Direct continuity','direct_continuity',percent]];
const descriptions={
  random_forest:'An ensemble of bootstrapped decision trees. Grouped validation selected this family for the final tuned classifier.',
  hist_gradient_boosting:'Histogram-based boosting fits successive shallow trees. This model had the lowest student-test log loss among the five baseline models.',
  logistic_regression:'A standardised linear model. It achieved the highest student-test Top-1 among the five baseline models.',
  gaussian_nb:'A Gaussian likelihood model that assumes conditionally independent features given the class.',
  knn:'A distance-weighted vote among neighbouring candidate feature vectors after scaling.',
};

export default function Reports({evidence}){
  const [selected,setSelected]=useState('random_forest');
  const model=evidence.models.find(row=>row.key===selected);
  const experiment=evidence.routeEvaluation;
  return <div className="view">
    <section className="intro"><div><h1>Model reports</h1><p>Five classifiers, one trip-grouped split, and ten held-out student-test trips.</p></div></section>
    <section className="panel comparison">
      <header className="panel-header"><div><h2>Classifier comparison</h2><p>Baseline models. Select any row for its full diagnostic report.</p></div></header>
      <div className="table-scroll"><table><thead><tr><th>Model</th><th>CV Macro F1</th><th>Student Top-1</th><th>Student Macro F1</th><th>Log loss</th></tr></thead><tbody>
        {evidence.models.map(row=><tr key={row.key} className={row.key===selected?'selected':''}><th><button className="row-select" aria-pressed={row.key===selected} onClick={()=>setSelected(row.key)}>{row.name}</button></th>
          <td>{fixed(row.cv_macro_f1)}</td><td>{percent(row.student_test.top1_accuracy)}</td><td>{fixed(row.student_test.macro_f1)}</td><td>{fixed(row.student_test.log_loss)}</td></tr>)}
      </tbody></table></div>
    </section>
    <section className="panel model-details" key={selected}>
      <header className="panel-header"><div><h2>{model.name}</h2><p>{model.owner}</p></div><span className="pill">{model.sampling.rows.toLocaleString()} fit rows</span></header>
      <p className="details-description">{descriptions[selected]}</p>
      <div className="model-kpis">
        <div><small>Macro precision</small><strong>{fixed(model.student_test.macro_precision)}</strong></div>
        <div><small>Macro recall</small><strong>{fixed(model.student_test.macro_recall)}</strong></div>
        <div><small>Candidate ROC AUC</small><strong>{fixed(model.candidate.roc_auc)}</strong></div>
        <div><small>Average precision</small><strong>{fixed(model.candidate.average_precision)}</strong></div>
        <div><small>Final fit on Mac</small><strong>{fixed(model.final_fit_seconds,3)} s</strong></div>
        <div><small>Test inference on Mac</small><strong>{fixed(model.student_test.inference_seconds,3)} s</strong></div>
      </div>
      <details><summary>Model parameters and validation folds</summary>
        <div className="folds">{model.folds.map((value,i)=><span key={i}>Fold {i+1}<b>{fixed(value)}</b></span>)}</div>
        <div className="parameter-grid">{Object.entries(model.configuration).map(([key,value])=><div key={key}><code>{key}</code><span>{String(value)}</span></div>)}</div>
      </details>
      <figure><img loading="lazy" src={model.chart} alt={model.name+' confusion matrix, ROC, precision–recall and cross-validation plots'}/><figcaption>Candidate diagnostics count one selected road per GPS observation. Route Macro F1 averages road-class scores across trips.</figcaption></figure>
    </section>
    <section className="panel comparison">
      <header className="panel-header"><div><h2>Route reconstruction comparison</h2><p>Ten student-test trips · {experiment.refined.observations.toLocaleString()} GPS observations</p></div></header>
      <div className="table-scroll"><table><thead><tr><th>Measure</th><th>Tuned pointwise</th><th>Original decoder</th><th>Refined decoder</th></tr></thead><tbody>
      {metrics.map(([title,key,format])=><tr key={key}><th>{title}</th>{['classifier','original','refined'].map(mode=><td key={mode}>{format(experiment[mode][key])}</td>)}</tr>)}
      </tbody></table></div>
      <p className="text-note">Decoder settings were chosen on training validation. Student-test results are development evidence. Official evaluator trips are separate.</p>
    </section>
  </div>;
}
