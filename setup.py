from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def download_requirements(file_path:str)->List[str]:
    requires=[]
    with open(file_path) as file_obj:
        requires=file_obj.readlines()
        requires=[require.replace("/","") for require in requires]
        if HYPEN_E_DOT in requires:
            requires.remove(HYPEN_E_DOT)
    return requires

setup(
    name='Air Pollution Prediction',
    version='0.0.1',
    author='Gautam',
    author_email='dashgautam331@gmail.com',
    install_requires=download_requirements('requirement.txt'),
    packages=find_packages()
)