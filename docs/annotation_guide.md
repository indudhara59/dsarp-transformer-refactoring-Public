# Annotation guide

Annotators independently complete `outputs/datasets/annotation_template.csv`. `is_applicable` is 0/1. `relevance_grade` is 0 invalid/harmful, 1 possible but low value, 2 useful, 3 highly useful, or 4 critical. Benefit and risk are continuous `[0,1]`. Reviewer IDs must be stable pseudonyms; notes justify uncertainty but are never inserted into model input.

Multiple reviews are preserved. Applicability uses majority vote, relevance the median, benefit/risk the mean, with reviewer count and disagreement recorded. Ties at 0.5 currently resolve applicable and should be adjudicated before final evaluation.

