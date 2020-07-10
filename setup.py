import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="open-competition", # Replace with your own username
    version="0.1",
    author="Example",
    author_email="author@example.com",
    description="A small example package",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
