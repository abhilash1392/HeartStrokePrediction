from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


models = {'logreg':LogisticRegression(),'decision_tree_gini':DecisionTreeClassifier(criterion='gini'),'decision_tree_entropy':DecisionTreeClassifier(criterion='entropy'),'xgb':XGBClassifier()}
