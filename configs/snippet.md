logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      name: test
      project: MINST_VAE
      offline: False
      save_dir: ./log