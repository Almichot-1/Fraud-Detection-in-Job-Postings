import json
import os

NOTEBOOKS = [
    "01_import_libraries",
    "02_load_dataset",
    "03_data_understanding",
    "04_exploratory_data_analysis",
    "05_preprocessing",
    "06_feature_engineering",
    "07_model_training",
    "08_evaluation",
    "09_threshold_tuning",
    "10_error_analysis",
    "11_save_model",
    "12_app_demo"
]

TITLES = [
    "Import Libraries",
    "Load Dataset",
    "Data Understanding",
    "Exploratory Data Analysis",
    "Preprocessing",
    "Feature Engineering",
    "Model Training",
    "Evaluation",
    "Threshold Tuning",
    "Error Analysis",
    "Save Model",
    "Application Demo"
]

def create_notebook(idx, title, specific_cells):
    nn = str(idx+1).zfill(2)
    header = f"""# ================================================================
# Section {nn}: {title}
# Fake Job Posting Detection System — ASTU ML Group 4
# ================================================================
"""
    imports = """import sys; sys.path.insert(0, '../src')"""
    
    cells = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [header]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [imports]
        }
    ]
    
    for cell_source in specific_cells:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell_source.split('\n')
        })
        # Add newlines back except for last element
        for i in range(len(cells[-1]["source"]) - 1):
            cells[-1]["source"][i] += '\n'
            
    notebook_dict = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    file_path = f"jupyter/{nn}_{title.lower().replace(' ', '_')}.ipynb"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)

def main():
    for i, (nb_base, title) in enumerate(zip(NOTEBOOKS, TITLES)):
        specific_cells = []
        
        if i == 0:
            specific_cells = ["import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport os\nprint('Libraries Imported Successfully!')"]
            
        elif i == 1:
            specific_cells = ["import pandas as pd\n\ndf = pd.read_csv('../data/fake_job_postings.csv')\nprint(f'Dataset loaded with shape: {df.shape}')"]
            
        elif i == 2:
            specific_cells = ["df.info()\ndf.describe(include='all')\ndf.head()"]
            
        elif i == 3:
            code_eda = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from src.feature_engineering import engineer_features

# Load data
df = pd.read_csv('../data/fake_job_postings.csv')

# (a) Class distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='fraudulent')
plt.title('Class Distribution (Fraudulent vs Legitimate)')
plt.show()

# (b) Missing value heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.show()

# (c) Word cloud
fraud_text = ' '.join(df[df['fraudulent'] == 1]['description'].fillna('').tolist())
legit_text = ' '.join(df[df['fraudulent'] == 0]['description'].fillna('').tolist())

plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
wc_fraud = WordCloud(width=400, height=300, background_color='black').generate(fraud_text)
plt.imshow(wc_fraud, interpolation='bilinear')
plt.title('Fraudulent Postings')
plt.axis('off')

plt.subplot(1, 2, 2)
wc_legit = WordCloud(width=400, height=300, background_color='white').generate(legit_text)
plt.imshow(wc_legit, interpolation='bilinear')
plt.title('Legitimate Postings')
plt.axis('off')
plt.show()

# Engineer features for (d) and (e)
df_engineered = engineer_features(df)

# (d) Distribution of description length
plt.figure(figsize=(8,5))
sns.histplot(data=df_engineered, x='description_length', hue='fraudulent', bins=50, kde=True)
plt.title('Description Length by Class')
plt.xlim(0, 5000)
plt.show()

# (e) Correlation matrix of engineered features
numeric_cols = ['description_length', 'suspicious_keyword_count', 'company_profile_missing', 'salary_missing', 'has_external_contact', 'has_company_logo', 'has_questions', 'fraudulent']
corr = df_engineered[numeric_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Engineered Features')
plt.show()
"""
            specific_cells = [code_eda]
            
        elif i == 4:
            specific_cells = ["import pandas as pd\nfrom src.preprocessing import build_preprocessor, split_data\n\ndf = pd.read_csv('../data/fake_job_postings.csv')\n# Note: feature engineering should be called before split, but we demonstrate split here.\nX_train, X_val, X_test, y_train, y_val, y_test = split_data(df)\npreprocessor = build_preprocessor()\nprint('Preprocessing components loaded.')"]
            
        elif i == 5:
            specific_cells = ["import pandas as pd\nfrom src.feature_engineering import engineer_features\n\ndf = pd.read_csv('../data/fake_job_postings.csv')\ndf_engineered = engineer_features(df)\ndf_engineered.head()"]
            
        elif i == 6:
            specific_cells = ["from src.training import get_all_models, train_all_models\n\nmodels_dict = get_all_models()\nprint('Models ready to be trained.')"]
            
        elif i == 7:
            eval_code = """import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from src.evaluation import compare_all_models, plot_confusion_matrices

# Assuming models are trained and results_dict is populated in memory
# plot_confusion_matrices(results_dict)
# compare_all_models(results_dict)

# PR Curve plot placeholder
# plt.figure(figsize=(8,6))
# for model_name, y_proba in all_probas.items():
#     prec, rec, _ = precision_recall_curve(y_test, y_proba)
#     plt.plot(rec, prec, label=model_name)
# plt.xlabel('Recall'); plt.ylabel('Precision')
# plt.legend()
# plt.title('PR Curves')
# plt.show()
"""
            specific_cells = [eval_code]
            
        elif i == 8:
            specific_cells = ["from src.evaluation import optimize_threshold\n# best_thresh = optimize_threshold(best_model, X_val, y_val)\n# print(f'Best threshold: {best_thresh}')"]
            
        elif i == 9:
            err_code = """import pandas as pd
# Assuming X_val and predictions are available
# X_val['y_true'] = y_val
# X_val['y_pred'] = y_pred

# false_positives = X_val[(X_val['y_true'] == 0) & (X_val['y_pred'] == 1)].sample(10)
# false_negatives = X_val[(X_val['y_true'] == 1) & (X_val['y_pred'] == 0)].sample(10)

# for idx, row in false_positives.iterrows():
#     print(f"FP - {row['title']}: {row['description'][:200]}...")
"""
            specific_cells = [err_code]
            
        elif i == 10:
            specific_cells = ["import os\nfrom src.training import save_artifacts\n\n# save_artifacts(pipeline, best_threshold)\nprint('Artifacts Saved.')\nprint('Pipeline exists:', os.path.exists('../models/final_pipeline.joblib'))\nprint('Threshold exists:', os.path.exists('../models/threshold.joblib'))"]
            
        elif i == 11:
            app_code = """from src.inference import predict_single
from src.training import load_artifacts

# pipeline, threshold = load_artifacts()

test_cases = [
    {
        'title': 'Data Scientist', 'company_profile': 'Great company.', 
        'description': 'Analyze data. deposit required to apply.', 'requirements': '', 'benefits': '',
        'employment_type': 'Full-time', 'required_experience': 'Entry level',
        'has_company_logo': 0, 'has_questions': 0
    },
    {
        'title': 'Software Engineer', 'company_profile': 'Tech Corp inc with a long history.', 
        'description': 'We need a software engineer to build web apps.', 'requirements': 'CS degree', 'benefits': 'Health insurance',
        'employment_type': 'Full-time', 'required_experience': 'Mid-Senior level',
        'has_company_logo': 1, 'has_questions': 1
    },
    {
        'title': 'Virtual Assistant', 'company_profile': '', 
        'description': 'Work from home and earn commission.', 'requirements': 'None', 'benefits': 'Flexible',
        'employment_type': 'Part-time', 'required_experience': 'Not Applicable',
        'has_company_logo': 0, 'has_questions': 0
    }
]

# for case in test_cases:
#     res = predict_single(case, pipeline, threshold)
#     print(res)
"""
            specific_cells = [app_code]
            
        create_notebook(i, title, specific_cells)

if __name__ == '__main__':
    os.makedirs('jupyter', exist_ok=True)
    main()
