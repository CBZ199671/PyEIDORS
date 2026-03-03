from setuptools import setup, Extension
setup(name="hello", ext_modules=[Extension("hello", ["hello.c"])])
