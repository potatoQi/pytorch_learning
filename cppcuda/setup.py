import glob
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

sources = glob.glob('*.cpp') + glob.glob('*.cu')    # 会返回当前目录下所有 .cpp 和 .cu 文件的名字的列表

setup(
    name='cppcuda',
    version='1.0',
    author='Error_666',
    author_email='qixingzhou1125@outlook.com',
    description='cppcuda',
    long_description='A package for cppcuda extension',
    ext_modules=[
        CUDAExtension(
            name='cppcuda',
            sources=sources,
            extra_compile_args={
                'cxx': ['-O2'],         # cpp O2 编译优化
                'nvcc': ['-O2']         # cuda O2 编译优化
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)