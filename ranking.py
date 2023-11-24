from sklearn.svm import SVC, LinearSVC
import pandas as pd
import logging
import numpy as np


class Ranker():

    def __init__(self, company, worker_pool):
        self.company = company
        self.worker_pool = worker_pool

    def hire(X, y, s):
        pass
    
    def get_workerpool_train_format(self):
        pass

    def get_employee_train_format(self, numpy=False, perceived=False, sensitive_attr='all', past=False):
        """ return X, s, y where y is the productivity """
        if perceived:
            attr = 'perceived_productivity'
        else:
            attr = 'productivity'

        # iterate over X, y, s dictionaries
        features = [e.features for e in self.company.employees.values()] + [m.features for m in self.company.managers.values()]
        features = pd.DataFrame(features, columns=self.company.datasource.feature_cols)
        sensitives = [e.sensitive_attributes for e in self.company.employees.values()] \
                + [m.sensitive_attributes for m in self.company.managers.values()]
        sensitives = pd.DataFrame(sensitives, columns=self.company.datasource.sensitive_cols)
        productivities = [getattr(e, attr) for e in self.company.employees.values()] \
                + [getattr(m, attr) for m in self.company.managers.values()]
        productivities = pd.DataFrame(productivities, columns=[self.company.datasource.y_col])

        # recover data from employees and managers who left
        if past is not False:
            logging.debug('Getting past data')
            past_features = [e.features for e in self.company.past_employees.values()] + [m.features for m in self.company.past_managers.values()]
            past_features = pd.DataFrame(past_features, columns=self.company.datasource.feature_cols)
            past_sensitives = [e.sensitive_attributes for e in self.company.past_employees.values()] \
                    + [m.sensitive_attributes for m in self.company.past_managers.values()]
            past_sensitives = pd.DataFrame(past_sensitives, columns=self.company.datasource.sensitive_cols)
            past_productivities = [getattr(e, attr) for e in self.company.past_employees.values()] \
                    + [getattr(m, attr) for m in self.company.past_managers.values()]
            past_productivities = pd.DataFrame(past_productivities, columns=[self.company.datasource.y_col])
            if past != 1:
                num = int(len(past_productivities) * past)
                logging.debug('Got {} examples from the past'.format(num))
                past_features = past_features.iloc[:num, :]
                past_sensitives = past_sensitives.iloc[:num, :]
                past_productivities = past_productivities.iloc[:num, :]
            features = pd.concat([features, past_features], axis=0)
            sensitives = pd.concat([sensitives, past_sensitives], axis=0)
            productivities = pd.concat([productivities, past_productivities], axis=0)

        # handle return values per function parameters
        if sensitive_attr != 'all':
            sensitives = sensitives.loc[:, sensitive_attr]
        if numpy:
            features, sensitives, productivities = np.array(features), np.array(sensitives), np.array(productivities)
        return features, sensitives, productivities
    

