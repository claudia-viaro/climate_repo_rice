## repo structure
RICE-N-exp/
│
├── environment.yaml               # Conda env definition
│
├── scripts/                       # Entrypoints and standalone scripts
│   ├── trainer.py
│   ├── run_cluster.sh                   # empty now
│   └── __init__.py
│
├── my_project/
│   ├── __init__.py
│   │
│   ├── analysis/                  # Empty for now
│   │   └── __init__.py
│   │
│   ├── configs/
│   │   ├── __init__.py
│   │   ├── other_yamls/
│   │   │   └── ...
│   │   ├── region_yamls/
│   │   │   └── ...
│   │   ├── envs/                  # scenario definitions, environment configs
│   │   │   ├── scenarios.py
│   │   │   ├── rice_env.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── training/
│   │   │   ├── torch_models_discrete.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── utils/
│   │   │   ├── fixed_paths.py
│   │   │   ├── create_submission_zip.py
│   │   │   ├── rice_rllib_discrete.yaml
│   │   │   └── __init__.py
│   │
│   └── __init__.py
|
├── outputs/               # runtime logs/checkpoints 
│   └── .gitignore         # empty to avoid cluttering git
│
│
└── README.md
└── requirements.txt
└── setup.cfg              # empty
└── pyproject.toml         # empty

