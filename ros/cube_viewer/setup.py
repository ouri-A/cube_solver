from setuptools import find_packages, setup

package_name = 'cube_viewer'

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
    description='3D view of the scanned cube that starts the solve and follows it move by move.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'viewer = cube_viewer.viewer:main',
        ],
    },
)
