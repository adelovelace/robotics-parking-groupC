from glob import glob
from setuptools import find_packages, setup

package_name = 'robomaster_example'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch') + glob(package_name + '/launch/*.launch')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Elia Cereda',
    maintainer_email='eliacereda@gmail.com',
    description='RoboMaster object-boundary exploration and parking pipeline.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Refactored nodes
            'vision_observer_node = robomaster_example.nodes.vision_observer_node:main',
            'map_node = robomaster_example.nodes.map_node:main',
            'slot_detector_node = robomaster_example.nodes.slot_detector_node:main',
            'mission_controller_node = robomaster_example.nodes.mission_controller_node:main',
            'visualization_node = robomaster_example.nodes.visualization_node:main',
            'controller_park4 = robomaster_example.nodes.controller_park4:main',

            # Backward-compatible alias: old launch files that call controller_node
            # will start the new mission controller.
            'controller_node = robomaster_example.nodes.mission_controller_node:main',
        ],
    },
)
