from setuptools import setup, find_packages

setup(
    name="plant_disease_detector",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'Flask==2.3.3',
        'torch==2.0.1',
        'torchvision==0.15.2',
        'Pillow==10.0.0',
        'numpy==1.24.3',
        'Werkzeug==2.3.7',
        'gunicorn==21.2.0',
        'python-dotenv==1.0.0',
        'requests==2.31.0'
    ],
    python_requires='>=3.8',
)
