![Confusion matrix](random-forest)

# TikTok Claim vs Opinion: Engagement Analysis and Classifier

## Question
Do claim-based videos get treated differently by the algorithm than opinion-based videos, and can we predict a video's type from its engagement?

## Data
19,382 videos, 12 columns (views, likes, shares, downloads, comments, duration, claim status).
298 rows had missing values and were removed, leaving 19,084. The top 1% of views and likes were then trimmed as outliers, leaving 18,733.

## Method
1. Cleaned the data and checked missing values.
2. Explored video counts, durations, and engagement with charts.
3. Tested whether view counts differ between the two groups (Welch's t-test).
4. Built features: engagement rate and log of views.
5. Trained logistic regression and random forest models using a 70/15/15 train/validation/test split.

## Results
- Claim videos averaged about 484,000 views. Opinion videos averaged about 5,000 (statistically significant).
- Random forest test accuracy: 99.47%.
- Logistic regression test accuracy: 98.90%.
- The most important feature was view count (about 91% of the random forest's importance).

## Limitations
Engagement, especially views, separates the two groups so sharply that the task is easy for a model. Accuracy this high mainly reflects that gap, not model sophistication.

## Files
- `analysis.ipynb`: full analysis
- `tiktok_dataset.csv`: dataset

## Tools
Python, pandas, scikit-learn, seaborn, matplotlib, scipy

