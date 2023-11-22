from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa import DataCollector
import logging
from datasources import GeneratorDataSource, DataSource

logger = logging.getLogger(__name__)

class Worker(Agent):
    def __init__(self, unique_id, model, features, productivity, sensitive_attributes):
        super().__init__(unique_id, model)
        self.features = features
        self.productivity = productivity
        self.perceived_productivity = self.calculate_perceived_productivity()
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
    def __init__(self, unique_id, model, features, productivity, sensitive_attributes):
        super().__init__(unique_id, model, features, productivity, sensitive_attributes)
        self.type = "employee"

    def step(self):
        # Define the employee's behavior per step here
        logger.debug('I am employee {}'.format(self.unique_id))
        logger.debug('My features are {}'.format(self.features))
        logger.debug('My productivity is {}. But nobody knows that.'.format(self.productivity))
        # leaving the company

class Manager(Worker):
    """ An agent representing a manager. """
    def __init__(self, unique_id, model, features, productivity, sensitive_attributes):
        super().__init__(unique_id, model, features, productivity, sensitive_attributes)
        self.type = "manager"

    def step(self):
        # Define the manager's behavior per step here
        logger.debug('I am manager {}'.format(self.unique_id))
        logger.debug('My features are {}'.format(self.features))
        logger.debug('My productivity is {}. But nobody knows that.'.format(self.productivity))
        # leaving the company

class CompanyModel(Model):
    """ A model representing a company with employees and managers. """
    def __init__(self, n_employees, n_managers, datasource: DataSource):
        self.schedule = RandomActivation(self)
        self.employees = []
        self.managers = []
        data = datasource.generate_dataset()
        features, sensitive_attrs = try_detect_sensitive_attrs(data)
        self.column_names = data.columns
        
        # Create employees
        for i in range(n_employees):
            features_i = features.iloc[i, :-1]
            productivity_i = features.iloc[i, -1] # assumption warning!
            sens_attr_i = sensitive_attrs.iloc[i, :]
            employee = Employee(i, self, features_i, productivity_i, sens_attr_i)
            self.employees.append(employee)
            self.schedule.add(employee)

        # Create managers
        for i in range(n_employees, n_employees + n_managers):
            features_i = features.iloc[i, :-1]
            productivity_i = features.iloc[i, -1]
            sens_attr_i = sensitive_attrs.iloc[i, :]
            manager = Manager(i, self, features_i, productivity_i, sens_attr_i)
            self.managers.append(manager)
            self.schedule.add(manager)

    def step(self):
        """ Advance the model by one step. """
        self.schedule.step()
        productivity = self.current_productivity()
        logger.info('Step {}; Productivity is {}'.format(self.schedule.steps, productivity))

    def current_productivity(self):
        return sum([e.productivity for e in self.employees] + [m.productivity for m in self.managers])

    def get_data_train_format():
        # return X, s, y
        # note that we have more than one s, often
        return []


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

    
