import argparse
import logging
from agents import CompanyModel
from datasources import GeneratorDataSource

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulation')
    parser.add_argument('-w', '--workers', type=int, default=100, help='Number of workers')
    parser.add_argument('-e', '--employees', type=int, default=10, help='Number of employees')
    parser.add_argument('-m', '--managers', type=int, default=5, help='Number of managers')
    parser.add_argument('-s', '--steps', type=int, default=10, help='Number of steps')
    parser.add_argument('--male-perc', type=float, default=0.5, help='Percentage of male people')
    parser.add_argument('--white-perc', type=float, default=0.5, help='Percentage of white people')
    parser.add_argument('--min-prod', type=float, default=1.0, help='Minimum productivity per agent')
    parser.add_argument('--max-prod', type=float, default=10.0, help='Maximum productivity per agent')
    parser.add_argument('--loglevel', default='WARNING', help='Set the logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--empl-leave-prob', type=float, default=0.1, help='Probability of employee leaving')
    parser.add_argument('--man-leave-prob', type=float, default=0.1, help='Probability of manager leaving')
    parser.add_argument('--past-data', type=float, default=0.0, help='Proportion of data from past employees to take')
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    # Instantiate data source
    gender_proportions = {'male': args.male_perc, 'female': 1-args.male_perc}
    ethnicity_proportions = {'black': 1-args.white_perc, 'white': args.white_perc}
    productivity_params = {'min': 1, 'max': 10}
    generator = GeneratorDataSource(100, gender_proportions, ethnicity_proportions, productivity_params)
    # Instantiate model
    model = CompanyModel(args.employees, args.managers, generator, employee_leaving_prob=args.empl_leave_prob,
                         manager_leaving_prob=args.man_leave_prob)
    for i in range(args.steps):  # Run for 10 steps
        model.step()
    # Show data
    data = model.datacollector.get_model_vars_dataframe()
    print(data)
    X, s, y = model.get_data_train_format(past=args.past_data, numpy=False, perceived=False)
    print(X)
    print(s)
    print(y)


