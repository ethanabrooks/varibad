import ast
import re
from pathlib import Path
import dataclasses
from rich.console import Console
from rich.table import Table
from dataclasses import fields
import config.config  # Assuming config.py has been imported


console = Console()


def extract_args_from_file(file_name):
    with open(file_name, "r") as f:
        tree = ast.parse(f.read())

    args = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
        ):
            arg_name = node.args[0].s if len(node.args) > 0 else None
            default_value = None
            help_str = None

            for keyword in node.keywords:
                if keyword.arg == "default":
                    if isinstance(keyword.value, ast.Constant):
                        default_value = keyword.value.value
                    elif isinstance(keyword.value, ast.List):
                        default_value = [
                            el.value
                            for el in keyword.value.elts
                            if isinstance(el, (ast.Constant, ast.Num))
                        ]  # We handle both constant and numbers for compatibility.
                    # You can handle other node types here if necessary

                if keyword.arg == "help":
                    help_str = (
                        keyword.value.s if isinstance(keyword.value, ast.Str) else None
                    )

            if arg_name:
                arg_name = arg_name.replace("--", "").replace("-", "_")
                args[arg_name] = {"default": default_value, "help": help_str}
    return args


def validate_args_with_config():
    console.print("[bold blue]Validating args with config.py[/bold blue]\n")

    discrepancies = []
    missing_arguments = []
    missing_classes = []

    for file_path in Path(".").rglob("args_*.py"):
        file_name = file_path.name
        class_name = file_name_to_class_name(file_name)
        config_definitions = extract_dataclass_definitions(class_name)

        if config_definitions is None:
            missing_classes.append(file_name)
            continue

        script_args = extract_args_from_file(file_path)
        for arg, details in script_args.items():
            if arg in config_definitions:
                if details["default"] != config_definitions[arg]["default"]:
                    discrepancies.append(
                        (
                            file_name,
                            class_name,
                            arg,
                            config_definitions[arg]["default"],
                            details["default"],
                        )
                    )
            else:
                missing_arguments.append((file_name, arg))

    if discrepancies:
        table = Table(title="Discrepancies Detected")
        table.add_column("Script Name", style="green")
        table.add_column("Class Name", style="green")
        table.add_column("Argument", style="green")
        table.add_column("Config Value", style="cyan")
        table.add_column("Script Value", style="cyan")
        for row in discrepancies:
            str_row = [str(item) for item in row]
            table.add_row(*str_row)
        console.print(table)

    if missing_arguments:
        table = Table(title="Missing Arguments in config.py")
        table.add_column("Script Name", style="magenta")
        table.add_column("Argument", style="magenta")
        for row in missing_arguments:
            table.add_row(*row)
        console.print(table)

    if missing_classes:
        table = Table(title="Missing Classes in config.py")
        table.add_column("Script Name", style="yellow")
        for row in missing_classes:
            table.add_row(row)
        console.print(table)

    console.print("[bold blue]Validation complete![/bold blue]")


def file_name_to_class_name(file_name):
    # Remove 'args_' prefix and '.py' suffix
    name = re.sub(r"args_|\.py", "", file_name)
    name = name.replace("pointrobot", "sparse_point_robot")
    name = name.replace("rl_", "r_l_")
    name = name.replace("_rl2", "_r_l_2")
    # Convert 'pointrobot_multitask' to 'PointRobotMultitask'
    class_name = "".join([word.capitalize() for word in name.split("_")])
    return class_name


def extract_dataclass_definitions(class_name):
    # Get the dataclass from the config module by name
    dataclass = getattr(config.config, class_name, None)
    if not dataclass:
        return
    definitions = {}
    for field_info in dataclass.__dataclass_fields__.values():
        default_value = None
        if (
            field_info.default_factory != dataclasses.MISSING
        ):  # Check if default_factory is used
            try:
                default_value = field_info.default_factory()  # Try calling the factory
            except Exception:
                # Could not produce the default value; might need special handling
                pass

        elif (
            field_info.default != dataclasses.MISSING
        ):  # Otherwise, just grab the default if it exists
            default_value = field_info.default

        definitions[field_info.name] = {
            "default": default_value,
            "type": field_info.type,
        }
    return definitions


validate_args_with_config()
