from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa import DataCollector
import argparse
import logging

logger = logging.getLogger(__name__)

class Worker(Agent):
    def __init__(self, unique_id, model, characteristics, actual_productivity):
        super().__init__(unique_id, model)
        self.characteristics = characteristics
        self.actual_productivity = actual_productivity
        self.perceived_productivity = self.calculate_perceived_productivity()

    def calculate_perceived_productivity(self):
        return 0

    def step(self):
        # Implement base worker behavior
        pass

class Employee(Worker):
    """ An agent representing an employee. """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, [0, 0], 0)
        self.type = "employee"

    def step(self):
        # Define the employee's behavior per step here
        logger.debug('I am employee {}'.format(self.unique_id))
        # leaving the company

class Manager(Worker):
    """ An agent representing a manager. """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model, [0, 0], 0)
        self.type = "manager"

    def step(self):
        # Define the manager's behavior per step here
        logger.debug('I am manager {}'.format(self.unique_id))
        # leaving the company

class CompanyModel(Model):
    """ A model representing a company with employees and managers. """
    def __init__(self, n_employees, n_managers):
        self.schedule = RandomActivation(self)
        self.employees = []
        self.managers = []
        
        # Create employees
        for i in range(n_employees):
            employee = Employee(i, self)
            self.employees.append(employee)
            self.schedule.add(employee)

        # Create managers
        for i in range(n_employees, n_employees + n_managers):
            manager = Manager(i, self)
            self.managers.append(manager)
            self.schedule.add(manager)

    def step(self):
        """ Advance the model by one step. """
        self.schedule.step()
        productivity = self.current_productivity()
        logger.info('Step {}; Productivity is {}'.format(self.schedule.steps, productivity))

    def current_productivity(self):
        return sum([e.actual_productivity for e in self.employees] + [m.actual_productivity for m in self.managers])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('-w', '--workers', type=int, default=100, help='Number of workers')
    parser.add_argument('-e', '--employees', type=int, default=10, help='Number of employees')
    parser.add_argument('-m', '--managers', type=int, default=5, help='Number of managers')
    parser.add_argument('-s', '--steps', type=int, default=10, help='Number of steps')
    parser.add_argument('--loglevel', default='INFO', help='Set the logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    # Instantiate and run the model
    model = CompanyModel(args.employees, args.managers)
    for i in range(10):  # Run for 10 steps
        model.step()


