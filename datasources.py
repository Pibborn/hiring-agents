import numpy as np
import pandas as pd
import random
import logging

logger = logging.getLogger()

class DataSource():

    def __init__():
        pass

    def generate_dataset():
        raise NotImplementedError()

    
class GeneratorDataSource(DataSource):
    def __init__(self, num_employees, gender_proportions, ethnicity_proportions, productivity_params):
        self.num_employees = num_employees
        self.gender_proportions = gender_proportions
        self.ethnicity_proportions = ethnicity_proportions
        self.productivity_params = productivity_params

    def generate_gender(self):
        genders = []
        for gender, proportion in self.gender_proportions.items():
            count = int(proportion * self.num_employees)
            genders.extend([gender] * count)
        random.shuffle(genders)
        return genders[:self.num_employees]

    def generate_ethnicity(self):
        ethnicities = []
        for ethnicity, proportion in self.ethnicity_proportions.items():
            count = int(proportion * self.num_employees)
            ethnicities.extend([ethnicity] * count)
        random.shuffle(ethnicities)
        return ethnicities[:self.num_employees]

    def generate_dataset(self):
        # Generate sensitive attributes for all employees at once
        # This part depends on how generate_sensitive_attributes is implemented
        genders = self.generate_gender()
        ethnicities = self.generate_ethnicity()
        print(len(genders))

        # Generate other attributes for all employees
        experiences = np.random.normal(5, 2, self.num_employees)  # Mean 5, SD 2
        skills = np.random.uniform(1, 10, self.num_employees)

        # Calculate productivity for all employees
        productivities = self.generate_productivity(experiences, skills)

        # Combine into dataset
        dataset = pd.DataFrame({
            'experience': experiences,
            'skill': skills,
            'gender': genders,
            'ethnicity': ethnicities,
            'productivity': productivities
        })
        return dataset

    def generate_productivity(self, experiences, skills):
        # Calculate productivity as a weighted sum of experiences and skills
        # Adjust the weights as needed to reflect their impact on productivity
        experience_weight = 0.5  # Example weight for experience
        skill_weight = 0.5       # Example weight for skill

        # Linear combination of experience and skill
        base_productivity = (experiences * experience_weight) + (skills * skill_weight)

        # Add random noise to introduce variability
        random_noise = np.random.normal(0, 1, self.num_employees)  # Gaussian noise

        # Final productivity
        return base_productivity + random_noise


if __name__ == '__main__':
    # Example usage
    num_employees = 100
    gender_proportions = {'male': 0.5, 'female': 0.5}
    ethnicity_proportions = {'white': 0.5, 'black': 0.25, 'asian': 0.25}
    productivity_params = {'min': 1.0, 'max': 10.0}

    generator = GeneratorDataSource(num_employees, gender_proportions, ethnicity_proportions, productivity_params)
    dataset = generator.generate_dataset()
    print(dataset)

