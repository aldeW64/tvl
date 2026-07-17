from setuptools import find_packages, setup

setup(
    name="tvl",
    version="0.1",
    packages=find_packages(
        include=[
            "tvl_flextok",
            "tvl_flextok.*",
            "tvl_enc",
            "tvl_enc.*",
            "tvl_llama",
            "tvl_llama.*",
        ]
    ),
    package_data={
        "tvl_flextok": ["configs/*.yaml"],
        "tvl_enc": ["data/*.png"],
    },
)
