from setuptools import find_packages, setup

package_name = 'cube_executor'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ouri-A',
    maintainer_email='ouri200425@gmail.com',
    description='Stand-in for the robot: carries out an Execute goal one move at a time.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'executor = cube_executor.executor:main',
        ],
    },
)
