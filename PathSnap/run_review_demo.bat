@echo off
python -m src.pathsnap.train --data-root "data\raw\Student folder" --trip-ids 00000026,00000039,00000018,00000014,00000022 --models logistic,naive_bayes,knn,random_forest,hist_gradient_boosting
pause
