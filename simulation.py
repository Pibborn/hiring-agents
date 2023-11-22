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
    parser.add_argument('--loglevel', default='INFO', help='Set the logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))
    # Instantiate data source
    gender_proportions = {'male': args.male_perc, 'female': 1-args.male_perc}
    ethnicity_proportions = {'black': 1-args.white_perc, 'white': args.white_perc}
    productivity_params = {'min': 1, 'max': 10}
    generator = GeneratorDataSource(100, gender_proportions, ethnicity_proportions, productivity_params)
    # Instantiate model
    model = CompanyModel(args.employees, args.managers, generator)
    for i in range(10):  # Run for 10 steps
        model.step()


