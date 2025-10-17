# Actions taken in the repo
1. Create folders for repo structure src/datasets, src/infer, src/models

2. Create a virtual enviroment:
    cd ~/Documents/HeritageArt-CNN-ViT-Hybrid
    python3 -m venv venv
    source venv/bin/activate

3. Created requirements.txt file and executed pip install -r requirements.txt

4. Install Tools for pre-commit
    pip install pre-commit black flake8 isort hydra-core

5. Created the file .pre-commit-config-yaml, pyproject.toml and .flake8

6. Run:
    pre-commit install
    pre-commit run --all-files

7. Created the yaml files in configs and commit chagnes

8. Organized experiments folder (2025-10-06):
    - Moved cuda-test.py, MMSeg-test.py, demo/image_demo.py → experiments/
    - Created experiments/README.md with usage instructions
    - Updated pre-commit hooks to exclude experiments/ from linting
    - Adjusted paths in moved scripts to work from new location

9. Used agent mode to build contenarized POC call poc-art-damage in experiments folder

# Next actions:

Implement src/datasets/artefact_dataset.py using the scaffold code from the plan

Write a test in tests/test_dataset.py