# Candidate-screening data

`candidate78_final_predictions.csv` contains the final D and P outputs for the
78 frozen electrolyte candidates. D scores are rank-normalized eight-member
fusion scores calculated within this candidate queue; P is the output of the
frozen single-LightGBM model. `candidate78_pareto_knee.csv` contains the eight
Pareto-nondominated candidates and the precomputed geometric knee distances.

The selection uses normalized benefit scores from D and P ranks, removes
Pareto-dominated candidates, and selects the frontier member with the largest
perpendicular distance to the chord joining the two frontier endpoints.
DMTMSA is the unique maximum-distance knee in the released table.
