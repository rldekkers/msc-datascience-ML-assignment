import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from lingua import Language, LanguageDetectorBuilder
detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")
test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")

# PREPROCESSING

# combine author's into string
train['author_combined'] = train['author'].apply(lambda x:'?'.join(map(str, x)) if x is not None else '')
test['author_combined'] = test['author'].apply(lambda x:'?'.join(map(str, x)) if x is not None else '')

# combine 'title' and 'abstract'
train['text_combined'] = train['title'] + train['abstract']
test['text_combined'] = test['title'] + test['abstract']

# find language of title
train['title_lang'] = [lang.name for lang in detector.detect_languages_in_parallel_of(train['title'])]
test['title_lang'] = [lang.name for lang in detector.detect_languages_in_parallel_of(test['title'])]

train, val = train_test_split(train, stratify=train['year'], random_state=123)

featurizer = ColumnTransformer([
    ("tfidf_authors", TfidfVectorizer(tokenizer=lambda text: text.split('?'), token_pattern=None), "author_combined"),
    ("tfidf_abstract", TfidfVectorizer(min_df=0.01, stop_words='english'), "abstract"),
    ("tfidf_title", TfidfVectorizer(min_df=0.01, stop_words='english'), "title"),
    ("tfidf_publisher", TfidfVectorizer(), "publisher"),
    ("tfidf_text", TfidfVectorizer(min_df=0.01, stop_words="english"), "text_combined"),
    ("ohe", OneHotEncoder(handle_unknown='ignore'), ["ENTRYTYPE", "publisher", "title_lang"])],
    remainder='drop'
)

# pipe = Pipeline([('featurizer', featurizer), ('ridge', Ridge(alpha=1))])
pipe = Pipeline([('featurizer', featurizer), 
                ('linear_csv', LinearSVC(C= 0.4, loss='squared_hinge', dual=False, verbose=2))])
# pipe = Pipeline([('featurizer', featurizer),
#                  ('logistic_reg', LogisticRegression(verbose=2,
#                                                     n_jobs=-1,
#                                                     max_iter=100))])
# pipe = Pipeline([('featurizer', featurizer), ('rfr', RandomForestRegressor(n_jobs=-1, verbose=10))])
# pipe = Pipeline([('featurizer', featurizer), ('sgd', SGDClassifier(verbose=10) )])

pipe.fit(train.drop('year', axis=1), train['year'].values)  

err_train = mean_absolute_error(train['year'].values, pipe.predict(train.drop('year', axis=1)))
err_val = mean_absolute_error(val['year'].values, pipe.predict(val.drop('year', axis=1)))
print()
print(f"Train MAE:       {err_train:.3f}")
print(f"Validation MAE:  {err_val:.3f}")
print()



# feature importance
# feat_imp = permutation_importance(pipe,
#                                   X = val.drop('year', axis=1),
#                                   y = val['year'].values)

# print(feat_imp)



# grid_search = GridSearchCV(pipe, param_grid = param_grid, cv=3, verbose=10, n_jobs=-1)
# grid_search.fit(train.drop('year', axis=1), train['year'].values
# print(grid_search.cv_results_)

pred = pipe.predict(test)
test['year'] = pred
test.to_json("predicted.json", orient='records', indent=2)



