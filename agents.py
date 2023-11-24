from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa import DataCollector
import logging
from datasources import GeneratorDataSource, DataSource
import random
import numpy as np
import pandas as pd
from ranking import Ranker

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
        pass


class WorkerPool():
    def __init__(self, datasource: DataSource, model: Model):
        self.datasource = datasource
        self.model = model
        self.workers = self.generate_workers()

    def generate_workers(self):
        self.data = self.datasource.generate_dataset()
        features, sensitive_attrs = try_detect_sensitive_attrs(self.data)
        self.num_workers = len(self.data)
        workers = {}
        for i in range(self.num_workers):
            features_i = features.iloc[i, :-1]
            productivity_i = features.iloc[i, -1] # assumption warning!
            sens_attr_i = sensitive_attrs.iloc[i, :]
            worker = Worker(i, self.model, features_i, productivity_i, sens_attr_i)
            workers[i] = worker
            # TODO add unemployed workers to schedule?
        return workers


class Employee(Worker):
    """ An agent representing an employee. """
    def __init__(self, unique_id, model, features, productivity, sensitive_attributes, leaving_probability,
                 step_hired=0):
        super().__init__(unique_id, model, features, productivity, sensitive_attributes)
        self.type = "employee"
        self.leaving_probability = leaving_probability
        self.step_hired = step_hired

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

    def promotion(self):
        logger.debug('Employee {} is getting promoted!'.format(self.unique_id))
        del self.model.employees[self.unique_id]
        self.model.past_employees[self.unique_id] = self
        manager_who_is_me = Manager(self.unique_id, self.model, self.features, self.productivity,
                                    self.sensitive_attributes, self.leaving_probability, step_hired=self.step_hired,
                                    step_promoted=self.model.schedule.steps)
        self.model.managers[self.unique_id] = manager_who_is_me


class Manager(Worker):
    """ An agent representing a manager. """
    def __init__(self, unique_id, model, features, productivity, sensitive_attributes, leaving_probability,
                 step_hired=0, step_promoted=-1):
        super().__init__(unique_id, model, features, productivity, sensitive_attributes)
        self.type = "manager"
        self.leaving_probability = leaving_probability
        self.step_hired = step_hired
        self.step_promoted = step_promoted

    def step(self):
        # Define the manager's behavior per step here
        logger.debug('I am manager {}'.format(self.unique_id))
        logger.debug('My features are {}'.format(self.features))
        logger.debug('My productivity is {}. But nobody knows that.'.format(self.productivity))
        check = random.random()
        # TODO: it would make more sense to move this into the model
        if check <= self.leaving_probability:
            logger.info('Manager {} leaving!'.format(self.unique_id))
            self.model.schedule.remove(self)
            del self.model.managers[self.unique_id]
            self.model.past_managers[self.unique_id] = self


class CompanyModel(Model):
    """ A model representing a company with employees and managers. """
    def __init__(self, n_employees, n_managers, datasource: DataSource,
                 employee_leaving_prob=0.1, manager_leaving_prob=0.1, gap=1):
        self.n_employees_start = n_employees
        self.n_managers_start = n_managers
        self.schedule = RandomActivation(self)
        self.employee_leaving_prob = employee_leaving_prob
        self.manager_leaving_prob = manager_leaving_prob
        self.employees = {}
        self.managers = {}
        self.id_counter = 0
        self.datacollector = DataCollector(
                model_reporters={"Firm Productivity": self.current_productivity}
                )
        # order of operations matters here -- quite messy
        self.datasource = datasource
        self.worker_pool = WorkerPool(self.datasource, self)
        self.do_initial_hires()
        features, sensitive_attrs = try_detect_sensitive_attrs(self.worker_pool.data)
        self.column_names = features.columns
        self.past_employees = {}
        self.past_managers = {}
        self.ranker = Ranker(self, self.worker_pool)
        self.gap = gap
        

    def do_initial_hires(self):
        # set worker pool -- this stays throughout the life of this company
        step_hired = 0
        # Create employees
        for i in range(self.n_employees_start):
            worker = self.worker_pool.workers[i]
            employee = Employee(i, self, worker.features, worker.productivity, worker.sensitive_attributes,
                                self.employee_leaving_prob, step_hired=step_hired)
            del self.worker_pool.workers[i]
            self.employees[i] = employee
            self.schedule.add(employee)

        # Create managers
        for i in range(self.n_employees_start, self.n_employees_start + self.n_managers_start):
            worker = self.worker_pool.workers[i]
            manager = Manager(i, self, worker.features, worker.productivity, worker.sensitive_attributes,
                              self.manager_leaving_prob, step_hired=step_hired)
            del self.worker_pool.workers[i]
            self.managers[i] = manager
            self.schedule.add(manager)

        # Ensure unique ids even after re-hiring
        self.id_counter = self.n_employees_start + self.n_managers_start


    def step(self):
        """ Advance the model by one step. """
        self.datacollector.collect(self)
        self.schedule.step()
        productivity = self.current_productivity()
        logger.info('Step {}; Productivity is {}'.format(self.schedule.steps, productivity))
        if productivity == 0:
            logger.warning('Productivity dropped to 0!')
        # hire if needed
        if self.should_we_hire():
            self.hire()
        if self.should_we_promote():
            self.promote()

    def hire(self, howmany):
        workers = ranker.get_top_candidates(howmany)
        for worker in workers:
            self.employees[worker.unique_id] = worker.am_hired()

    def promote(self):
        promotion_manager = random.choice(list(self.managers.items()))
        promoted_employee = promotion_manager.select_new_manager()
        new_manager = Manager(promoted_employee.unique_id, self, promoted_employee.features,
                              promoted_employee.productivity, promoted_employee.sensitive_attributes, 
                              self.manager_leaving_prob)
        del self.employees[promoted_employee.unique_id]
        self.managers[new_manager.unique_id] = new_manager


    def should_we_hire(self):
        return len(self.employees) <= self.n_employees_start - self.gap

    def should_we_promote(self):
        return len(self.managers) <= self.n_managers_start - self.gap

    def current_productivity(self):
        """ Returns productivity of firm. """
        return sum([e.productivity for e in self.employees.values()] + [m.productivity for m in self.managers.values()])


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


