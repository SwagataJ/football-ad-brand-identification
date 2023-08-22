import subprocess
import os


def _clone_dependency_and_download_requirements(dependency):
    """Clones a dependency from a git repository and installs its setup."""
    dependency_name = dependency.split("@")[0]
    dependency_url = dependency.split("@")[1]
    #os.chdir(dependency_name)
    subprocess.run(["git", "clone", dependency_url])
    subprocess.run(["pip", "install", "-r", "{}/requirements.txt".format(dependency_name)])
    #subprocess.run(["touch", "{}/__init__.py".format(dependency_name)])


def _clone_dependency(dependency):
    """Clones a dependency from a git repository."""
    dependency_name = dependency.split("@")[0]
    dependency_url = dependency.split("@")[1]
    subprocess.run(["git", "clone", dependency_url])
    os.chdir(dependency_name)

    
def _download_dependency_pip(dependency):
    """Downloads a dependency using pip."""
    dependency_name = dependency.split("@")[0]
    subprocess.run(["pip", "install", dependency_name], check=True)


def _download_weights(dependency_url):
    """Downloads the model weights."""
    subprocess.run(["wget", dependency_url, "-O", "glass-text-spotting/models/glass_textocr.pth"])


def main():
    """Installs the dependencies."""
    _download_dependency_pip("torch")
    _download_dependency_pip("torchvision")
    _download_dependency_pip("PyYAML")
    _download_dependency_pip("wcmatch")
    _download_dependency_pip("cython")
    _download_dependency_pip("git+https://github.com/facebookresearch/detectron2.git@v0.6")
    _clone_dependency("U-2-Net@https://github.com/NathanUA/U-2-Net.git")
    _clone_dependency_and_download_requirements("glass-text-spotting@https://github.com/SwagataJ/glass-text-spotting")
    _download_weights("https://glass-text-spotting.s3.eu-west-1.amazonaws.com/models/glass_250k_icdar15_fintune.pth")


if __name__ == "__main__":
    main()
