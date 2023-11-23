from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa import DataCollector
import logging
from datasources import GeneratorDataSource, DataSource
import random
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class Worker(Agent):
    def __init__(self, unique_id, model, features, productivity, sensitive_attributes):
        super().__init__(unique_id, model)
        self.features = features
        self.productivity = productivity
        self.perceived_productivity = productivity - 2 # TODO
        self.sensitive_attributes = sensitive_attributes

    def calculate_perceived_productivity(self):
        # TODO
        return 0

    def step(self):
        # Implement base worker behavior
        # TODO?
        pass


class Employee(Worker):
    """ An agent representing an employee. """
    def __init__(self, unique_id, model, features, productivity, sensitive_attributes, leaving_probability):
        super().__init__(unique_id, model, features, productivity, sensitive_attributes)
        self.type = "employee"
        self.leaving_probability = leaving_probability

    def step(self):
        # Define the employee's behavior per step here
        logger.debug('I am employee {}'.format(self.unique_id))
        logger.debug('My features are {}'.format(self.features))
        logger.debug('My productivity is {}. But nobody knows that.'.format(self.productivity))
        # leaving the company
        check = random.random()
        if check <= self.leaving_probability:
            logger.info('Employee {} leaving!'.format(self.unique_id))
            self.model.schedule.remove(self)
            del self.model.employees[self.unique_id]
            self.model.past_employees[self.unique_id] = self


class Manager(Worker):
    """ An agent representing a manager. """
    def __init__(self, unique_id, model, features, productivity, sensitive_attributes, leaving_probability):
        super().__init__(unique_id, model, features, productivity, sensitive_attributes)
        self.type = "manager"
        self.leaving_probability = leaving_probability

    def step(self):
        # Define the manager's behavior per step here
        logger.debug('I am manager {}'.format(self.unique_id))
        logger.debug('My features are {}'.format(self.features))
        logger.debug('My productivity is {}. But nobody knows that.'.format(self.productivity))
        check = random.random()
        if check <= self.leaving_probability:
            logger.info('Employee {} leaving!'.format(self.unique_id))
            self.model.schedule.remove(self)
            del self.model.managers[self.unique_id]
            self.model.past_managers[self.unique_id] = self


class CompanyModel(Model):
    """ A model representing a company with employees and managers. """
    def __init__(self, n_employees, n_managers, datasource: DataSource, employee_leaving_prob=0.1,
                 manager_leaving_prob=0.1):
        self.schedule = RandomActivation(self)
        self.employees = {}
        self.managers = {}
        self.id_counter = 0
        self.datacollector = DataCollector(
                model_reporters={"Firm Productivity": self.current_productivity}
            )
        self.datasource = datasource
        data = datasource.generate_dataset()
        features, sensitive_attrs = try_detect_sensitive_attrs(data)
        self.column_names = data.columns
        self.past_employees = {}
        self.past_managers = {}
        
        # Create employees
        for i in range(n_employees):
            features_i = features.iloc[i, :-1]
            productivity_i = features.iloc[i, -1] # assumption warning!
            sens_attr_i = sensitive_attrs.iloc[i, :]
            employee = Employee(i, self, features_i, productivity_i, sens_attr_i, employee_leaving_prob)
            self.employees[i] = employee
            self.schedule.add(employee)

        # Create managers
        for i in range(n_employees, n_employees + n_managers):
            features_i = features.iloc[i, :-1]
            productivity_i = features.iloc[i, -1]
            sens_attr_i = sensitive_attrs.iloc[i, :]
            manager = Manager(i, self, features_i, productivity_i, sens_attr_i, manager_leaving_prob)
            self.managers[i] = manager
            self.schedule.add(manager)
        
        # Ensure unique ids even after re-hiring
        self.id_counter = n_employees + n_managers

    def step(self):
        """ Advance the model by one step. """
        self.datacollector.collect(self)
        self.schedule.step()
        productivity = self.current_productivity()
        logger.info('Step {}; Productivity is {}'.format(self.schedule.steps, productivity))
        if productivity == 0:
            logger.warning('Productivity dropped to 0!')
        # hire if needed
        # TODO


    def current_productivity(self):
        """ Returns productivity of firm. """
        return sum([e.productivity for e in self.employees.values()] + [m.productivity for m in self.managers.values()])

    def get_data_train_format_actual(self):
        # note that we have more than one s, often
        features = [e.features() for e in self.employees.values()] + [m.features() for m in self.managers.values()]
        sensitives = [e.sensitive_attributes() for e in self.employees.values()] \
                     + [m.sensitive_attributes() for m in self.managers.values()]
        productivities = [e.productivity for e in self.employees.values()] \
                         + [m.productivity for m in self.managers.values()]
        return np.array(features), np.array(sensitives), np.array(productivities)

    def get_data_train_format(self, numpy=False, perceived=False, sensitive_attr='all', past=False):
        """ return X, s, y where y is the productivity """
        if perceived:
            attr = 'perceived_productivity'
        else:
            attr = 'productivity'
        features = [e.features for e in self.employees.values()] + [m.features for m in self.managers.values()]
        features = pd.DataFrame(features, columns=self.datasource.feature_cols)
        sensitives = [e.sensitive_attributes for e in self.employees.values()] \
                     + [m.sensitive_attributes for m in self.managers.values()]
        sensitives = pd.DataFrame(sensitives, columns=self.datasource.sensitive_cols)
        productivities = [getattr(e, attr) for e in self.employees.values()] \
                         + [getattr(m, attr) for m in self.managers.values()]
        productivities = pd.DataFrame(productivities, columns=[self.datasource.y_col])
        if past is not False:
            logging.debug('Getting past data')
            past_features = [e.features for e in self.past_employees.values()] + [m.features for m in self.past_managers.values()]
            past_features = pd.DataFrame(past_features, columns=self.datasource.feature_cols)
            past_sensitives = [e.sensitive_attributes for e in self.past_employees.values()] \
                              + [m.sensitive_attributes for m in self.past_managers.values()]
            past_sensitives = pd.DataFrame(past_sensitives, columns=self.datasource.sensitive_cols)
            past_productivities = [getattr(e, attr) for e in self.past_employees.values()] \
                                  + [getattr(m, attr) for m in self.past_managers.values()]
            past_productivities = pd.DataFrame(past_productivities, columns=[self.datasource.y_col])
            if past != 1:
                num = int(len(past_productivities) * past)
                logging.debug('Got {} examples from the past'.format(num))
                past_features = past_features.iloc[:num, :]
                past_sensitives = past_sensitives.iloc[:num, :]
                past_productivities = past_productivities.iloc[:num, :]
            features = pd.concat([features, past_features], axis=0)
            sensitives = pd.concat([sensitives, past_sensitives], axis=0)
            productivities = pd.concat([productivities, past_productivities], axis=0)


        if sensitive_attr != 'all':
            sensitives = sensitives.loc[:, sensitive_attr]
        if numpy:
            features, sensitives, productivities = np.array(features), np.array(sensitives), np.array(productivities)
        return features, sensitives, productivities



    def get_data_train_format_perceived(self):
        """ return X, s, y where y is the perceived productivity """
        # note that we have more than one s, often
        

def get_column_index(df, colname):
    try:
        return df.columns.get_loc(colname)
    except KeyError:
        return -1

def try_detect_sensitive_attrs(df):
    sens_attr_idx = []
    for sens_attr_name in ['gender', 'ethnicity']:
        idx = get_column_index(df, sens_attr_name)
        if idx != -1:
            sens_attr_idx.append(idx)
    return df.iloc[:, [i for i in range(df.shape[1]) if i not in sens_attr_idx]], df.iloc[:, sens_attr_idx]

    
