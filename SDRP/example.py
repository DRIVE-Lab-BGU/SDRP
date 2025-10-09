
import sys
from SDRP.ExampleManager import ExampleManager
import os


def main(domain, instance, config, episodes=100, step=10):
    manager = ExampleManager(domain, instance, config, episodes, step)
    manager.run_example()
    base_path = manager.get_base_path()
    log_file = os.path.join(base_path, 'logs', f"log_{instance[0:-5]}_sd = 1.csv")
    manager.log(log_file)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print('python example.py <domain> <instance> <config> [<episodes=100>] [<step=10>]')
        exit(1)
    kwargs = {'domain': args[0], 'instance': args[1], 'config': args[2]}
    if len(args) >= 4: kwargs['episodes'] = int(args[3])
    if len(args) >= 5: kwargs['step'] = int(args[4])
    main(**kwargs)