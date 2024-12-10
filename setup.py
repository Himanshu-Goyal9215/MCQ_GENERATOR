from setuptools import find_packages,setup

setup(
    name='MCQ_GENERATOR',
    version='0.0.1',
    author='HIMANSHU GOYAL',
    author_email='goyalhimanshu096@gmail.com',
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
)