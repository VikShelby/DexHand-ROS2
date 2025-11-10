from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'dexhand_ai'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'config'), 
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'torchvision',
        'transformers',
        'opencv-python',
        'ultralytics',
        'mediapipe',
        'pyyaml'
    ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='AI vision and learning for DexHand V1',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ai_gesture_bridge = dexhand_ai.ai_gesture_bridge:main',
            'learning_collector = dexhand_ai.learning_collector:main',
            'deploy_policy = dexhand_ai.deploy_policy:main',
        ],
    },
)