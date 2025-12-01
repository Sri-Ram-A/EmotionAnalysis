from pygments import highlight
from pygments.lexers import YamlLexer
from pygments.formatters import Terminal256Formatter
import yaml
import subprocess

def print_yaml(dictionary):
    yaml_str = yaml.dump(dictionary, sort_keys=False, default_flow_style=False)
    # Strong bright colors for dark background
    formatter = Terminal256Formatter(style="monokai")  
    print(highlight(yaml_str, YamlLexer(), formatter))

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Command failed")
    print(result.stdout)