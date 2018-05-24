from setuptools import setup

setup(
    name='lsd',
    version='0.1',
    description='Experimental framework for NN Learning on Synthetic Data (LSD) and monitoring with replica computations',
    url='https://github.com/marylou-gabrie/learning-synthetic-data',
    author='Marylou Gabrie, Andre Manoel, Florent Krzakala, Gabriel Samain',
    author_email='marylou.gabrie@gmail.com',
    license='MIT',
    packages=['lsd', 'lsd.teacher_student', 'lsd.decoder_encoder'],
    zip_safe=False
)
