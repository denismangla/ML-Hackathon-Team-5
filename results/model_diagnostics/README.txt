PathSnap Team 5, 11:30 model metrics

Each model uses saved predictions from the same 10 student-test trips.
The confusion matrix treats the selected candidate as positive and all other candidates as negative.
ROC and precision-recall use probabilities normalised within each observation.
Student Top-1 is GPS-point accuracy. Student Macro F1 averages road-class Macro F1 across trips.
A reference omitted by candidate generation creates no positive candidate row and still counts as wrong in point-level Top-1.
The three fold scores come from whole-trip validation within training.
