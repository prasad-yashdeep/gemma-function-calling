from setuptools import setup, find_packages

setup(
    name="gemma-sdk",
    version="0.1.0",
    description="Python SDK for Gemma LLM models with multi-turn, multi-API function calling capabilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Gemma Functional calling SDK Team",
    author_email="yashdeep18121@iiitd.ac.in",
    url="https://github.com/prasad-yashdeep/gemma-function-calling",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "pydantic>=2.0.0",
        "requests>=2.28.0",
        "openapi-parser>=0.5.0",
        "pyyaml>=6.0",
        "pillow>=9.0.0",  # For image API support
        "numpy>=1.24.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
