from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        equirements = [req.replace('\n', '') for req in requirements]
        
    return requirements

setup(
    name='Diamond_Price_Prediction',
    version='0.0.1',
    author='Aman Kumar',
    author_email= 'amankumarpanika446@gmail.com',
    install_requires= get_requirements('requirements.txt'),
    packages=find_packages(),
)