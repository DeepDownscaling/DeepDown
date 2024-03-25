from omegaconf import OmegaConf, DictConfig, ListConfig
from pathlib import Path
from rich import get_console
from rich.style import Style
from rich.tree import Tree
import typing

"""
Class to handle the configuration, such as path to data and directories.
"""


class Config:

    def __init__(self, cli_args):
        # Load options from default config file
        self.config = OmegaConf.load('../config.default.yaml')

        # Merge options from custom config file
        if cli_args.config_file:
            self.config = OmegaConf.merge(self.config,
                                          OmegaConf.load(cli_args.config_file))
        elif Path('config.yaml').exists():
            self.config = OmegaConf.merge(self.config,
                                          OmegaConf.load('config.yaml'))
        elif Path('../config.yaml').exists():
            self.config = OmegaConf.merge(self.config,
                                          OmegaConf.load('../config.yaml'))

        # Merge options from CLI
        cli_args_list = [f'{k}={v}' for k, v in cli_args.__dict__.items() if
                         v is not None]
        self.config = OmegaConf.merge(self.config,
                                      OmegaConf.from_cli(cli_args_list))

    def print(self) -> None:
        """Print content of given config using Rich library and its tree structure."""

        def walk_config(tree: Tree, config: DictConfig):
            """Recursive function to accumulate branch."""
            for group_name, group_option in config.items():
                if isinstance(group_option, dict):
                    branch = tree.add(str(group_name),
                                      style=Style(color='yellow', bold=True))
                    walk_config(branch, group_option)
                elif isinstance(group_option, ListConfig):
                    if not group_option:
                        tree.add(f'{group_name}: []',
                                 style=Style(color='yellow', bold=True))
                    else:
                        tree.add(f'{str(group_name)}: {group_option}',
                                 style=Style(color='yellow', bold=True))
                else:
                    if group_name == '_target_':
                        tree.add(f'{str(group_name)}: {group_option}',
                                 style=Style(color='white', italic=True, bold=True))
                    else:
                        tree.add(f'{str(group_name)}: {group_option}',
                                 style=Style(color='yellow', bold=True))

        tree = Tree(
            ':deciduous_tree: Configuration Tree ',
            style=Style(color='white', bold=True, encircle=True),
            guide_style=Style(color='bright_green', bold=True),
            expanded=True,
            highlight=True,
        )
        walk_config(tree, self.config)
        get_console().print(tree)

    def get(self):
        """Return the configuration."""
        return self.config
