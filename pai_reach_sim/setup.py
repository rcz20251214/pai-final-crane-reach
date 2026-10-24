from glob import glob

from setuptools import setup

package_name = 'pai_reach_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pai-student',
    maintainer_email='noreply@example.com',
    description='CRANE+ V2 reach task in Gazebo with an overhead camera',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'randomize_cube = pai_reach_sim.cube:main',
            'scripted_demo = pai_reach_sim.scripted_demo:main',
            'evaluate = pai_reach_sim.evaluate:main',
        ],
    },
)
